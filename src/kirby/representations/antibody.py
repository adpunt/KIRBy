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
        np.ndarray: (n_sequences, 1024) embeddings (AntiBERTa2 hidden_size)
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
        light_sequences: Light chain sequences (optional). If provided,
            heavy and light embeddings are concatenated (1536D for seq-codings).
        output_type:
            - 'seq-codings': 768D per sequence (1536D if light_sequences provided)
            - 'res-codings': 768D per residue, mean-pooled to per-sequence
        device: 'cpu' or 'cuda'

    Returns:
        np.ndarray: Embeddings (768D heavy-only or 1536D paired)
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

        if light_sequences is not None:
            # Also get light chain and concatenate (1536D)
            light_ablang = ablang.pretrained("light")
            light_ablang.freeze()
            light_embs = light_ablang(light_sequences, mode='seqcoding')
            return np.concatenate([heavy_embs, light_embs], axis=1)

        return heavy_embs

    elif output_type == 'res-codings':
        # Per-residue embeddings, mean-pooled
        heavy_embs = heavy_ablang(heavy_sequences, mode='rescoding')

        embeddings = []
        for emb in heavy_embs:
            embeddings.append(emb.mean(axis=0))

        return np.array(embeddings)

    else:
        raise ValueError(f"Unknown output_type: {output_type}. Must be 'seq-codings' or 'res-codings'.")


def create_esm2_embeddings(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    model_name: str = 'facebook/esm2_t33_650M_UR50D',
    device: str = 'cpu',
    batch_size: int = 8,
) -> np.ndarray:
    """
    Extract ESM-2 embeddings for antibody sequences.

    ESM-2 (Lin et al., Science 2023) is the most widely used general protein
    language model. Trained on 250M proteins from UniRef. The 650M variant
    produces 1280D embeddings and is the standard baseline for protein tasks.

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences. If provided,
            heavy and light embeddings are concatenated (2560D).
        model_name: HuggingFace model name. Common variants:
            - 'facebook/esm2_t33_650M_UR50D' (1280D, recommended)
            - 'facebook/esm2_t30_150M_UR50D' (640D, faster)
            - 'facebook/esm2_t36_3B_UR50D' (2560D, best quality)
        device: 'cpu' or 'cuda'
        batch_size: Number of sequences per batch

    Returns:
        np.ndarray: (n_sequences, 1280) or (n_sequences, 2560) if paired
    """
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        raise ImportError(
            "ESM-2 requires transformers: pip install transformers"
        )

    print(f"Loading ESM-2 ({model_name.split('/')[-1]})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()

    def _embed_sequences(sequences: List[str]) -> np.ndarray:
        all_embeddings = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = [s.upper().strip() for s in sequences[start:start + batch_size]]
                inputs = tokenizer(
                    batch, return_tensors='pt', padding=True,
                    truncation=True, max_length=1024
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                # Mean pool excluding special tokens (BOS at 0, EOS at end)
                attention_mask = inputs['attention_mask']
                # Zero out BOS and EOS positions
                attention_mask[:, 0] = 0
                for j in range(len(batch)):
                    last_tok = attention_mask[j].sum().item()
                    if last_tok > 0:
                        attention_mask[j, int(last_tok)] = 0
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1)
                mean_emb = sum_embeddings / count
                all_embeddings.append(mean_emb.cpu().numpy())

                if start + batch_size < len(sequences):
                    print(f"    Progress: {min(start + batch_size, len(sequences))}/{len(sequences)}")
        return np.concatenate(all_embeddings, axis=0)

    print(f"  Embedding {len(heavy_sequences)} heavy chain sequences...")
    heavy_embs = _embed_sequences(heavy_sequences)

    if light_sequences is not None:
        print(f"  Embedding {len(light_sequences)} light chain sequences...")
        light_embs = _embed_sequences(light_sequences)
        result = np.concatenate([heavy_embs, light_embs], axis=1)
        print(f"  Generated ESM-2 embeddings: {result.shape}")
        return result

    print(f"  Generated ESM-2 embeddings: {heavy_embs.shape}")
    return heavy_embs


def create_esmc_embeddings(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    model_name: str = 'esmc_600m',
    device: str = 'cpu',
) -> np.ndarray:
    """
    Extract ESM Cambrian (ESM-C) embeddings for antibody sequences.

    ESM-C (EvolutionaryScale, 2024) is much more efficient than ESM-2.
    The 600M variant matches ESM-2 3B quality at a fraction of the cost.

    Requires: pip install esm (v3+, from EvolutionaryScale)

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences. If provided,
            heavy and light embeddings are concatenated.
        model_name: 'esmc_300m' or 'esmc_600m' (recommended)
        device: 'cpu' or 'cuda'

    Returns:
        np.ndarray: (n_sequences, 1152) for 600M, or doubled if paired
    """
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein
    except ImportError:
        raise ImportError(
            "ESM-C requires the EvolutionaryScale esm package (v3+):\n"
            "  pip install esm\n"
            "Note: This is different from the older fair-esm package."
        )

    print(f"Loading ESM-C ({model_name})...")
    model = ESMC.from_pretrained(model_name).to(device).eval()

    def _embed_sequences(sequences: List[str]) -> np.ndarray:
        all_embeddings = []
        with torch.no_grad():
            for i, seq in enumerate(sequences):
                protein = ESMProtein(sequence=seq.upper().strip())
                protein_tensor = model.encode(protein)
                output = model.forward(protein_tensor)
                # output.embeddings: (1, seq_len, hidden_dim)
                emb = output.embeddings[0].mean(dim=0).cpu().numpy()
                all_embeddings.append(emb)
                if (i + 1) % 50 == 0:
                    print(f"    Progress: {i+1}/{len(sequences)}")
        return np.array(all_embeddings)

    print(f"  Embedding {len(heavy_sequences)} heavy chain sequences...")
    heavy_embs = _embed_sequences(heavy_sequences)

    if light_sequences is not None:
        print(f"  Embedding {len(light_sequences)} light chain sequences...")
        light_embs = _embed_sequences(light_sequences)
        result = np.concatenate([heavy_embs, light_embs], axis=1)
        print(f"  Generated ESM-C embeddings: {result.shape}")
        return result

    print(f"  Generated ESM-C embeddings: {heavy_embs.shape}")
    return heavy_embs


def create_igbert_embeddings(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    device: str = 'cpu',
    batch_size: int = 8,
) -> np.ndarray:
    """
    Extract IgBERT embeddings for antibody sequences.

    IgBERT (Kenlay et al., PLOS Comp Bio 2024) is the largest antibody-specific
    LM with paired sequence support. Two-stage pretraining on 2B unpaired +
    2M paired sequences from OAS. SOTA on sequence recovery, expression, binding.

    For paired sequences, heavy and light are encoded together with [SEP],
    leveraging cross-chain attention learned during paired pretraining.

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences. If provided,
            uses paired encoding (recommended, exploits cross-chain attention).
        device: 'cpu' or 'cuda'
        batch_size: Number of sequences per batch

    Returns:
        np.ndarray: (n_sequences, 1024)
    """
    try:
        from transformers import BertTokenizer, BertModel
    except ImportError:
        raise ImportError(
            "IgBERT requires transformers: pip install transformers"
        )

    print(f"Loading IgBERT (Exscientia/IgBert)...")
    tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert")
    model = BertModel.from_pretrained("Exscientia/IgBert").to(device).eval()

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(heavy_sequences), batch_size):
            end = min(start + batch_size, len(heavy_sequences))
            batch_heavy = [' '.join(list(s.upper().strip())) for s in heavy_sequences[start:end]]

            if light_sequences is not None:
                # Paired encoding: [CLS] heavy [SEP] light [SEP]
                batch_light = [' '.join(list(s.upper().strip())) for s in light_sequences[start:end]]
                inputs = tokenizer(
                    batch_heavy, batch_light,
                    return_tensors='pt', padding=True,
                    truncation=True, max_length=512
                )
            else:
                # Heavy-only
                inputs = tokenizer(
                    batch_heavy, return_tensors='pt', padding=True,
                    truncation=True, max_length=512
                )

            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Mean pool over non-padding tokens (excluding special tokens)
            attention_mask = inputs['attention_mask'].clone()
            attention_mask[:, 0] = 0  # Exclude [CLS]
            # Exclude [SEP] tokens
            for j in range(end - start):
                sep_positions = (inputs['input_ids'][j] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                for sep_pos in sep_positions:
                    attention_mask[j, sep_pos] = 0

            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            batch_emb = (sum_emb / count).cpu().numpy()
            all_embeddings.append(batch_emb)

            if end < len(heavy_sequences):
                print(f"    Progress: {end}/{len(heavy_sequences)}")

    result = np.concatenate(all_embeddings, axis=0)
    mode = "paired" if light_sequences is not None else "heavy-only"
    print(f"  Generated IgBERT embeddings ({mode}): {result.shape}")
    return result


def create_prot_t5_embeddings(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    device: str = 'cpu',
    batch_size: int = 4,
) -> np.ndarray:
    """
    Extract ProtT5-XL embeddings for antibody sequences.

    ProtT5-XL (Elnaggar et al., IEEE TPAMI 2022) is a T5 encoder-decoder
    trained on 2.1B proteins from BFD. Produces 1024D per-residue embeddings
    from the encoder. Basis for IgT5 and ABodyBuilder3.

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences. If provided,
            heavy and light embeddings are concatenated (2048D).
        device: 'cpu' or 'cuda'
        batch_size: Number of sequences per batch (ProtT5 is memory-hungry)

    Returns:
        np.ndarray: (n_sequences, 1024) or (n_sequences, 2048) if paired
    """
    try:
        from transformers import T5Tokenizer, T5EncoderModel
    except ImportError:
        raise ImportError(
            "ProtT5-XL requires transformers + sentencepiece:\n"
            "  pip install transformers sentencepiece"
        )

    import re as _re

    print(f"Loading ProtT5-XL (Rostlab/prot_t5_xl_bfd)...")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_bfd", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_bfd"
    ).to(device).eval()

    def _embed_sequences(sequences: List[str]) -> np.ndarray:
        all_embeddings = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = [
                    _re.sub(r'[UZOB]', 'X', ' '.join(list(s.upper().strip())))
                    for s in sequences[start:start + batch_size]
                ]
                inputs = tokenizer(
                    batch, return_tensors='pt', padding=True,
                    truncation=True, max_length=1024, add_special_tokens=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                # Mean pool over sequence length (use attention_mask)
                attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                sum_emb = (outputs.last_hidden_state * attention_mask).sum(dim=1)
                count = attention_mask.sum(dim=1).clamp(min=1)
                batch_emb = (sum_emb / count).cpu().numpy()
                all_embeddings.append(batch_emb)

                end = min(start + batch_size, len(sequences))
                if end < len(sequences):
                    print(f"    Progress: {end}/{len(sequences)}")
        return np.concatenate(all_embeddings, axis=0)

    print(f"  Embedding {len(heavy_sequences)} heavy chain sequences...")
    heavy_embs = _embed_sequences(heavy_sequences)

    if light_sequences is not None:
        print(f"  Embedding {len(light_sequences)} light chain sequences...")
        light_embs = _embed_sequences(light_sequences)
        result = np.concatenate([heavy_embs, light_embs], axis=1)
        print(f"  Generated ProtT5-XL embeddings: {result.shape}")
        return result

    print(f"  Generated ProtT5-XL embeddings: {heavy_embs.shape}")
    return heavy_embs


def create_ab_roberta_embeddings(
    sequences: List[str],
    device: str = 'cpu',
    batch_size: int = 16,
) -> np.ndarray:
    """
    Extract Ab-RoBERTa embeddings for antibody sequences.

    Ab-RoBERTa (Huh & Lee, 2025) is a RoBERTa model trained on 402M antibody
    sequences from OAS with a SAA (subword amino acid) tokenizer that
    separates germline V gene families in embedding space.

    Single-chain model (processes heavy or light independently).

    Args:
        sequences: List of antibody sequences (heavy or light)
        device: 'cpu' or 'cuda'
        batch_size: Number of sequences per batch

    Returns:
        np.ndarray: (n_sequences, 768) embeddings
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "Ab-RoBERTa requires transformers: pip install transformers"
        )

    print(f"Loading Ab-RoBERTa (mogam-ai/Ab-RoBERTa)...")
    tokenizer = AutoTokenizer.from_pretrained("mogam-ai/Ab-RoBERTa")
    model = AutoModel.from_pretrained("mogam-ai/Ab-RoBERTa").to(device).eval()

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = [s.upper().strip() for s in sequences[start:start + batch_size]]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=150
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Mean pool using attention mask
            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
            sum_emb = (outputs.last_hidden_state * attention_mask).sum(dim=1)
            count = attention_mask.sum(dim=1).clamp(min=1)
            batch_emb = (sum_emb / count).cpu().numpy()
            all_embeddings.append(batch_emb)

            end = min(start + batch_size, len(sequences))
            if end < len(sequences):
                print(f"    Progress: {end}/{len(sequences)}")

    result = np.concatenate(all_embeddings, axis=0)
    print(f"  Generated Ab-RoBERTa embeddings: {result.shape}")
    return result


def create_balm_embeddings(
    sequences: List[str],
    balm_repo_path: str = './BALM',
    device: str = 'cpu',
    batch_size: int = 8,
) -> np.ndarray:
    """
    Extract BALM embeddings for antibody sequences.

    BALM (Chen et al., Briefings in Bioinformatics 2024) is a 150M parameter
    antibody LM with ESM-2-like architecture plus antibody-aware positional
    embeddings and adaptive masking. Trained on 336M sequences clustered at
    40% identity.

    BALM is NOT on HuggingFace. Requires manual setup:
      1. git clone https://github.com/BEAM-Labs/BALM.git
      2. Download weights from Google Drive (link in repo README)
      3. Pass balm_repo_path pointing to the cloned repo

    Uses ESM tokenizer vocabulary with custom model architecture.

    Args:
        sequences: List of antibody sequences (heavy or light)
        balm_repo_path: Path to cloned BALM repo containing
            modeling_balm.py, tokenizer/, and pretrained-BALM/
        device: 'cpu' or 'cuda'
        batch_size: Number of sequences per batch

    Returns:
        np.ndarray: (n_sequences, 1024) embeddings
    """
    import sys
    import os

    balm_repo_path = os.path.abspath(balm_repo_path)
    if not os.path.isdir(balm_repo_path):
        raise FileNotFoundError(
            f"BALM repo not found at: {balm_repo_path}\n"
            "BALM requires manual setup:\n"
            "  1. git clone https://github.com/BEAM-Labs/BALM.git\n"
            "  2. Download weights from Google Drive (see repo README)\n"
            "  3. Pass balm_repo_path='path/to/BALM'"
        )

    # Add repo to path for custom modules
    if balm_repo_path not in sys.path:
        sys.path.insert(0, balm_repo_path)

    # Patch transformers compatibility: BALM's modeling_balm.py imports
    # find_pruneable_heads_and_indices and prune_linear_layer from
    # transformers.modeling_utils, but they moved to pytorch_utils.
    import transformers.modeling_utils as _mu
    for _fn_name in ('find_pruneable_heads_and_indices', 'prune_linear_layer'):
        if not hasattr(_mu, _fn_name):
            setattr(_mu, _fn_name, getattr(__import__('transformers.pytorch_utils', fromlist=[_fn_name]), _fn_name))

    try:
        from modeling_balm import BALMForMaskedLM
        from transformers import EsmTokenizer
    except ImportError as e:
        raise ImportError(
            f"Failed to import BALM modules from {balm_repo_path}: {e}\n"
            "Ensure the repo contains modeling_balm.py and ba_position_embedding.py"
        )

    tokenizer_path = os.path.join(balm_repo_path, 'tokenizer')
    weights_path = os.path.join(balm_repo_path, 'pretrained-BALM')

    if not os.path.isdir(weights_path):
        raise FileNotFoundError(
            f"BALM weights not found at: {weights_path}\n"
            "Download from Google Drive (see repo README) and place in pretrained-BALM/"
        )

    print(f"Loading BALM from {balm_repo_path}...")
    tokenizer = EsmTokenizer.from_pretrained(
        tokenizer_path, do_lower_case=False, model_max_length=168
    )
    model = BALMForMaskedLM.from_pretrained(weights_path).to(device).eval()

    # Extract the base model (without MLM head) for embeddings
    base_model = model.esm if hasattr(model, 'esm') else model

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = [s.upper().strip() for s in sequences[start:start + batch_size]]
            inputs = tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=168
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = base_model(**inputs)
            # Mean pool using attention mask
            hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
            sum_emb = (hidden * attention_mask).sum(dim=1)
            count = attention_mask.sum(dim=1).clamp(min=1)
            batch_emb = (sum_emb / count).cpu().numpy()
            all_embeddings.append(batch_emb)

            end = min(start + batch_size, len(sequences))
            if end < len(sequences):
                print(f"    Progress: {end}/{len(sequences)}")

    result = np.concatenate(all_embeddings, axis=0)
    print(f"  Generated BALM embeddings: {result.shape}")
    return result


def create_sapiens_embeddings(
    sequences: List[str],
    chain_type: str = 'heavy',
    device: str = 'cpu',
) -> np.ndarray:
    """
    Extract Sapiens embeddings for antibody sequences.

    Sapiens (Prihoda et al., mAbs 2022, part of BioPhi) is a tiny 5-layer
    BERT model (128D) trained on 20M human H + 19M human L sequences.
    Designed for humanization but produces useful compact embeddings.

    Args:
        sequences: List of antibody sequences
        chain_type: 'heavy' or 'light' (selects model)
        device: 'cpu' or 'cuda'

    Returns:
        np.ndarray: (n_sequences, 128) embeddings
    """
    try:
        from sapiens.sapiens import predict_scores
        from sapiens.sapiens import load_model as sapiens_load_model
    except ImportError:
        raise ImportError(
            "Sapiens requires BioPhi: pip install biophi\n"
            "Or: pip install sapiens"
        )

    print(f"Loading Sapiens ({chain_type} chain)...")

    # Sapiens internally uses a small BERT model
    # We extract embeddings by running the model and getting hidden states
    model_type = 'H' if chain_type == 'heavy' else 'L'
    sapiens_model = sapiens_load_model(model_type)

    # Sapiens models are small BERT wrappers — extract the underlying model
    # and run forward pass to get hidden states
    embeddings = []
    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        try:
            # Use the predict_scores API to get per-residue scores
            # Then extract the hidden representation
            scores = predict_scores(seq, model_type=model_type)
            # scores is (seq_len, vocab_size) — use the log-probabilities
            # as a sequence-level feature via mean pooling
            emb = np.mean(scores, axis=0)
            embeddings.append(emb)
        except Exception:
            # Fallback: zero vector
            embeddings.append(np.zeros(20))

        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(sequences)}")

    result = np.array(embeddings)
    print(f"  Generated Sapiens features: {result.shape}")
    return result


def create_pigen_embeddings(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Extract p-IgGen embeddings for antibody sequences.

    p-IgGen (Frith et al., Bioinformatics 2024) is a GPT-2-based autoregressive
    transformer (17.3M params, 768D) from OPIG trained to generate paired
    H+L sequences. We extract hidden states as embeddings.

    HuggingFace: ollieturnbull/p-IgGen

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences
        device: 'cpu' or 'cuda'

    Returns:
        np.ndarray: (n_sequences, 768)
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "p-IgGen requires transformers: pip install transformers"
        )

    print(f"Loading p-IgGen (ollieturnbull/p-IgGen)...")
    tokenizer = AutoTokenizer.from_pretrained("ollieturnbull/p-IgGen")
    model = AutoModelForCausalLM.from_pretrained(
        "ollieturnbull/p-IgGen"
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    n = len(heavy_sequences)

    with torch.no_grad():
        for i in range(n):
            heavy = heavy_sequences[i].upper().strip()
            if light_sequences is not None:
                light = light_sequences[i].upper().strip()
                # p-IgGen uses [SEP] between chains
                seq_input = f"{heavy}[SEP]{light}"
            else:
                seq_input = heavy

            inputs = tokenizer(
                seq_input, return_tensors='pt',
                truncation=True, max_length=512
            )
            # GPTNeoX doesn't accept token_type_ids
            inputs.pop('token_type_ids', None)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            # Extract last hidden state from the last transformer layer
            last_hidden = outputs.hidden_states[-1][0]  # (seq_len, 768)
            emb = last_hidden.mean(dim=0).cpu().numpy()
            embeddings.append(emb)

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{n}")

    result = np.array(embeddings)
    mode = "paired" if light_sequences is not None else "heavy-only"
    print(f"  Generated p-IgGen embeddings ({mode}): {result.shape}")
    return result


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
        features['global'] = base_embeddings

    if 'cdr' in scales:
        raise NotImplementedError(
            "CDR-scale embeddings not implemented in create_multiscale_embeddings. "
            "Use create_cdr_stratified_embeddings() instead."
        )

    if 'residue_stats' in scales:
        raise NotImplementedError(
            "Residue-stats scale not implemented. Requires per-residue embeddings "
            "as input (base_embeddings must be 3D: n_seqs x seq_len x embed_dim)."
        )

    return features


# ============================================================================
# PART 4B: CLASSICAL SEQUENCE REPRESENTATIONS
# ============================================================================

def create_kmer_frequency(
    sequences: List[str],
    k: int = 2,
) -> np.ndarray:
    """
    Compute k-mer (dipeptide/tripeptide) frequency features.

    Captures local sequence motifs without requiring alignment. Dipeptides
    (k=2) encode pairwise residue preferences; tripeptides (k=3) capture
    longer-range patterns.

    Args:
        sequences: List of amino acid sequences (heavy, light, or concatenated)
        k: k-mer size. 2 → 400D (20²), 3 → 8000D (20³). Default 2.

    Returns:
        np.ndarray: (n_sequences, 20^k) k-mer frequency matrix
    """
    aas = STANDARD_AAS
    n_aas = len(aas)
    dim = n_aas ** k
    aa_to_idx = {aa: i for i, aa in enumerate(aas)}

    n = len(sequences)
    features = np.zeros((n, dim), dtype=np.float64)

    print(f"Computing {k}-mer frequency features for {n} sequences (dim={dim})...")

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if len(seq) < k:
            continue

        count = 0
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            # Compute index: e.g. for k=2, idx = aa1_idx * 20 + aa2_idx
            idx = 0
            valid = True
            for c, char in enumerate(kmer):
                if char not in aa_to_idx:
                    valid = False
                    break
                idx += aa_to_idx[char] * (n_aas ** (k - 1 - c))
            if valid:
                features[i, idx] += 1
                count += 1

        if count > 0:
            features[i] /= count

    print(f"  Generated {k}-mer features: ({n}, {dim})")
    return features


# Amino acid groupings for physicochemical profile
_AA_GROUPS = {
    'aromatic': set('FWYH'),
    'aliphatic': set('AVILM'),
    'positive': set('RKH'),
    'negative': set('DE'),
    'polar': set('STNQ'),
    'nonpolar': set('AVILMFWPG'),
    'tiny': set('AGS'),
    'small': set('AGSTCDNPV'),
    'cysteine': set('C'),
}


def create_physicochemical_profile(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute comprehensive physicochemical profile from sequence.

    Uses AA_PROPERTIES (6 properties: hydrophobicity, charge, mw, volume,
    polarity, flexibility) with 4 summary statistics each (mean, std, min,
    max), plus amino acid group composition fractions. Interpretable and
    complementary to learned embeddings.

    Features per chain (34D):
        0-3:   hydrophobicity (mean, std, min, max)
        4-7:   charge (mean, std, min, max)
        8-11:  molecular weight (mean, std, min, max)
        12-15: volume (mean, std, min, max)
        16-19: polarity (mean, std, min, max)
        20-23: flexibility (mean, std, min, max)
        24: fraction aromatic  (F,W,Y,H)
        25: fraction aliphatic (A,V,I,L,M)
        26: fraction positive  (R,K,H)
        27: fraction negative  (D,E)
        28: fraction polar     (S,T,N,Q)
        29: fraction nonpolar  (A,V,I,L,M,F,W,P,G)
        30: fraction tiny      (A,G,S)
        31: fraction small     (A,G,S,T,C,D,N,P,V)
        32: fraction cysteine  (C)
        33: sequence length (normalized by 150)

    Heavy-only: 34D (light columns zero-padded to 68D).
    Paired: 68D (34 heavy + 34 light).

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences

    Returns:
        np.ndarray: (n_sequences, 68) physicochemical profile
    """
    n = len(heavy_sequences)
    features = np.zeros((n, 68), dtype=np.float64)
    prop_keys = ['hydrophobicity', 'charge', 'mw', 'volume', 'polarity', 'flexibility']
    group_names = list(_AA_GROUPS.keys())

    def _compute_chain_features(seq: str) -> np.ndarray:
        """Compute 34D feature vector for a single chain."""
        out = np.zeros(34, dtype=np.float64)
        seq = seq.upper().strip()
        if not seq:
            return out

        valid_aas = [aa for aa in seq if aa in AA_PROPERTIES]
        if not valid_aas:
            return out

        # 6 properties × 4 stats = 24D
        for p_idx, prop in enumerate(prop_keys):
            vals = np.array([AA_PROPERTIES[aa][prop] for aa in valid_aas])
            out[p_idx * 4] = vals.mean()
            out[p_idx * 4 + 1] = vals.std()
            out[p_idx * 4 + 2] = vals.min()
            out[p_idx * 4 + 3] = vals.max()

        # 9 group fractions + length = 10D
        seq_len = len(valid_aas)
        for g_idx, group in enumerate(group_names):
            group_set = _AA_GROUPS[group]
            out[24 + g_idx] = sum(1 for aa in valid_aas if aa in group_set) / seq_len

        # Normalized length
        out[33] = seq_len / 150.0

        return out

    print(f"Computing physicochemical profile for {n} antibodies...")

    for i in range(n):
        features[i, :34] = _compute_chain_features(heavy_sequences[i])
        if light_sequences is not None:
            features[i, 34:] = _compute_chain_features(light_sequences[i])

    print(f"  Generated physicochemical profile: ({n}, 68)")
    return features


def create_imgt_onehot(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Create full IMGT-numbered one-hot encoding (fixed-length).

    Numbers each sequence with ANARCI (IMGT scheme), then one-hot encodes
    each of 128 positions (1-128) × 20 amino acids. Provides alignment-free
    fixed-length representation that preserves positional information.

    FLAb2 (2025) showed IMGT one-hot matches billion-parameter PLMs given
    enough training data, making this a strong non-learned baseline.

    Heavy-only: 2560D (128 × 20).
    Paired: 5120D (2560 heavy + 2560 light).

    Requires ANARCI for IMGT numbering.

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences

    Returns:
        np.ndarray: (n_sequences, 5120) IMGT one-hot features
    """
    try:
        from ..datasets.sabdab import extract_cdr_sequences
    except ImportError:
        raise ImportError(
            "ANARCI required for IMGT one-hot encoding. "
            "Install: conda install -c bioconda anarci"
        )

    n = len(heavy_sequences)
    # Always allocate 5120D (heavy + light), light = zeros if not provided
    features = np.zeros((n, 5120), dtype=np.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    def _encode_chain(sequence: str, chain_type: str) -> np.ndarray:
        """One-hot encode 128 IMGT positions → 2560D."""
        out = np.zeros(2560, dtype=np.float32)
        try:
            result = extract_cdr_sequences(
                sequence, numbering='imgt', chain_type=chain_type
            )
            if result is None:
                return out

            for pos, aa in result['numbered_seq']:
                # Only use integer positions 1-128
                if isinstance(pos, int) and 1 <= pos <= 128:
                    aa = aa.upper()
                    if aa in aa_to_idx:
                        offset = (pos - 1) * 20 + aa_to_idx[aa]
                        out[offset] = 1.0
        except Exception:
            pass
        return out

    print(f"Computing IMGT one-hot encoding for {n} antibodies...")

    for i in range(n):
        features[i, :2560] = _encode_chain(heavy_sequences[i].strip(), 'heavy')
        if light_sequences is not None:
            features[i, 2560:] = _encode_chain(light_sequences[i].strip(), 'light')

    n_nonzero = np.count_nonzero(features.sum(axis=1))
    print(f"  Generated IMGT one-hot: ({n}, 5120), {n_nonzero}/{n} sequences numbered")
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
# PART 8: ADDITIONAL AUGMENTATION FEATURES
# ============================================================================

# Standard BLOSUM62 substitution matrix (symmetric, 20x20 for standard AAs)
# Order: A R N D C Q E G H I L K M F P S T W Y V
BLOSUM62_MATRIX = {
    'A': { 'A': 4,'R':-1,'N':-2,'D':-2,'C': 0,'Q':-1,'E':-1,'G': 0,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S': 1,'T': 0,'W':-3,'Y':-2,'V': 0},
    'R': { 'A':-1,'R': 5,'N': 0,'D':-2,'C':-3,'Q': 1,'E': 0,'G':-2,'H': 0,'I':-3,'L':-2,'K': 2,'M':-1,'F':-3,'P':-2,'S':-1,'T':-1,'W':-3,'Y':-2,'V':-3},
    'N': { 'A':-2,'R': 0,'N': 6,'D': 1,'C':-3,'Q': 0,'E': 0,'G': 0,'H': 1,'I':-3,'L':-3,'K': 0,'M':-2,'F':-3,'P':-2,'S': 1,'T': 0,'W':-4,'Y':-2,'V':-3},
    'D': { 'A':-2,'R':-2,'N': 1,'D': 6,'C':-3,'Q': 0,'E': 2,'G':-1,'H':-1,'I':-3,'L':-4,'K':-1,'M':-3,'F':-3,'P':-1,'S': 0,'T':-1,'W':-4,'Y':-3,'V':-3},
    'C': { 'A': 0,'R':-3,'N':-3,'D':-3,'C': 9,'Q':-3,'E':-4,'G':-3,'H':-3,'I':-1,'L':-1,'K':-3,'M':-1,'F':-2,'P':-3,'S':-1,'T':-1,'W':-2,'Y':-2,'V':-1},
    'Q': { 'A':-1,'R': 1,'N': 0,'D': 0,'C':-3,'Q': 5,'E': 2,'G':-2,'H': 0,'I':-3,'L':-2,'K': 1,'M': 0,'F':-3,'P':-1,'S': 0,'T':-1,'W':-2,'Y':-1,'V':-2},
    'E': { 'A':-1,'R': 0,'N': 0,'D': 2,'C':-4,'Q': 2,'E': 5,'G':-2,'H': 0,'I':-3,'L':-3,'K': 1,'M':-2,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'G': { 'A': 0,'R':-2,'N': 0,'D':-1,'C':-3,'Q':-2,'E':-2,'G': 6,'H':-2,'I':-4,'L':-4,'K':-2,'M':-3,'F':-3,'P':-2,'S': 0,'T':-2,'W':-2,'Y':-3,'V':-3},
    'H': { 'A':-2,'R': 0,'N': 1,'D':-1,'C':-3,'Q': 0,'E': 0,'G':-2,'H': 8,'I':-3,'L':-3,'K':-1,'M':-2,'F':-1,'P':-2,'S':-1,'T':-2,'W':-2,'Y': 2,'V':-3},
    'I': { 'A':-1,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-3,'E':-3,'G':-4,'H':-3,'I': 4,'L': 2,'K':-3,'M': 1,'F': 0,'P':-3,'S':-2,'T':-1,'W':-3,'Y':-1,'V': 3},
    'L': { 'A':-1,'R':-2,'N':-3,'D':-4,'C':-1,'Q':-2,'E':-3,'G':-4,'H':-3,'I': 2,'L': 4,'K':-2,'M': 2,'F': 0,'P':-3,'S':-2,'T':-1,'W':-2,'Y':-1,'V': 1},
    'K': { 'A':-1,'R': 2,'N': 0,'D':-1,'C':-3,'Q': 1,'E': 1,'G':-2,'H':-1,'I':-3,'L':-2,'K': 5,'M':-1,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'M': { 'A':-1,'R':-1,'N':-2,'D':-3,'C':-1,'Q': 0,'E':-2,'G':-3,'H':-2,'I': 1,'L': 2,'K':-1,'M': 5,'F': 0,'P':-2,'S':-1,'T':-1,'W':-1,'Y':-1,'V': 1},
    'F': { 'A':-2,'R':-3,'N':-3,'D':-3,'C':-2,'Q':-3,'E':-3,'G':-3,'H':-1,'I': 0,'L': 0,'K':-3,'M': 0,'F': 6,'P':-4,'S':-2,'T':-2,'W': 1,'Y': 3,'V':-1},
    'P': { 'A':-1,'R':-2,'N':-2,'D':-1,'C':-3,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-3,'L':-3,'K':-1,'M':-2,'F':-4,'P': 7,'S':-1,'T':-1,'W':-4,'Y':-3,'V':-2},
    'S': { 'A': 1,'R':-1,'N': 1,'D': 0,'C':-1,'Q': 0,'E': 0,'G': 0,'H':-1,'I':-2,'L':-2,'K': 0,'M':-1,'F':-2,'P':-1,'S': 4,'T': 1,'W':-3,'Y':-2,'V':-2},
    'T': { 'A': 0,'R':-1,'N': 0,'D':-1,'C':-1,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S': 1,'T': 5,'W':-2,'Y':-2,'V': 0},
    'W': { 'A':-3,'R':-3,'N':-4,'D':-4,'C':-2,'Q':-2,'E':-3,'G':-2,'H':-2,'I':-3,'L':-2,'K':-3,'M':-1,'F': 1,'P':-4,'S':-3,'T':-2,'W':11,'Y': 2,'V':-3},
    'Y': { 'A':-2,'R':-2,'N':-2,'D':-3,'C':-2,'Q':-1,'E':-2,'G':-3,'H': 2,'I':-1,'L':-1,'K':-2,'M':-1,'F': 3,'P':-3,'S':-2,'T':-2,'W': 2,'Y': 7,'V':-1},
    'V': { 'A': 0,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-2,'E':-2,'G':-3,'H':-3,'I': 3,'L': 1,'K':-2,'M': 1,'F':-1,'P':-2,'S':-2,'T': 0,'W':-3,'Y':-1,'V': 4},
}

STANDARD_AAS = 'ACDEFGHIKLMNPQRSTVWY'


def compute_aac_features(
    sequences: List[str],
) -> np.ndarray:
    """
    Compute amino acid composition (AAC) features.

    For each sequence, counts the frequency (proportion) of each of the 20
    standard amino acids. Order: A C D E F G H I K L M N P Q R S T V W Y.

    Args:
        sequences: List of amino acid sequences (heavy, light, or concatenated)

    Returns:
        np.ndarray: (n_sequences, 20) amino acid frequency matrix
    """
    n = len(sequences)
    features = np.zeros((n, 20), dtype=np.float64)
    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if not seq:
            continue
        for aa in seq:
            if aa in aa_to_idx:
                features[i, aa_to_idx[aa]] += 1
        seq_len = sum(1 for aa in seq if aa in aa_to_idx)
        if seq_len > 0:
            features[i] /= seq_len

    print(f"  Generated AAC features: ({n}, 20)")
    return features


def compute_blosum62_features(
    sequences: List[str],
) -> np.ndarray:
    """
    Compute mean-pooled BLOSUM62 encoding features.

    Each residue is encoded as its row in the BLOSUM62 substitution matrix
    (a 20D vector capturing evolutionary substitution preferences). The
    per-residue vectors are mean-pooled to produce a fixed-length 20D vector
    per sequence.

    Args:
        sequences: List of amino acid sequences

    Returns:
        np.ndarray: (n_sequences, 20) BLOSUM62-encoded features
    """
    n = len(sequences)
    features = np.zeros((n, 20), dtype=np.float64)
    aa_order = list(STANDARD_AAS)

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if not seq:
            continue

        residue_vectors = []
        for aa in seq:
            if aa in BLOSUM62_MATRIX:
                row = BLOSUM62_MATRIX[aa]
                vec = np.array([row[target_aa] for target_aa in aa_order], dtype=np.float64)
                residue_vectors.append(vec)

        if residue_vectors:
            features[i] = np.mean(residue_vectors, axis=0)

    print(f"  Generated BLOSUM62 features: ({n}, 20)")
    return features


def compute_cdr_length_features(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute CDR length features using ANARCI numbering.

    CDR-H3 length is the single most predictive structural feature for
    antibody binding — it determines the shape of the antigen-binding loop.

    Features (7 if paired, 4 if heavy-only):
        0: cdr_h1_length
        1: cdr_h2_length
        2: cdr_h3_length   (most important)
        3: vh_length        (total VH length)
        4: cdr_l1_length    (0 if no light chain)
        5: cdr_l2_length    (0 if no light chain)
        6: cdr_l3_length    (0 if no light chain)

    Falls back to heuristic CDR-H3 estimation if ANARCI is not installed.

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences

    Returns:
        np.ndarray: (n_sequences, 7) CDR length features
    """
    n = len(heavy_sequences)
    features = np.zeros((n, 7), dtype=np.float64)

    # Try ANARCI for proper numbering
    try:
        from ..datasets.sabdab import extract_cdr_sequences
        use_anarci = True
    except (ImportError, Exception):
        use_anarci = False
        warnings.warn(
            "ANARCI not available. Using VH length only (CDR lengths set to 0). "
            "Install ANARCI for proper CDR extraction: conda install -c bioconda anarci"
        )

    print(f"Computing CDR length features for {n} antibodies "
          f"(ANARCI={'yes' if use_anarci else 'no'})...")

    for i in range(n):
        heavy = heavy_sequences[i].strip()
        features[i, 3] = len(heavy)  # VH length always available

        if use_anarci:
            try:
                cdrs = extract_cdr_sequences(heavy, numbering='imgt', chain_type='heavy')
                if cdrs is not None:
                    features[i, 0] = len(cdrs.get('cdr1', ''))
                    features[i, 1] = len(cdrs.get('cdr2', ''))
                    features[i, 2] = len(cdrs.get('cdr3', ''))
            except Exception:
                pass

        if light_sequences is not None:
            light = light_sequences[i].strip()
            if use_anarci:
                try:
                    cdrs_l = extract_cdr_sequences(light, numbering='imgt', chain_type='light')
                    if cdrs_l is not None:
                        features[i, 4] = len(cdrs_l.get('cdr1', ''))
                        features[i, 5] = len(cdrs_l.get('cdr2', ''))
                        features[i, 6] = len(cdrs_l.get('cdr3', ''))
                except Exception:
                    pass

    print(f"  Generated CDR length features: ({n}, 7)")
    if use_anarci:
        h3_lengths = features[:, 2]
        valid = h3_lengths > 0
        if valid.any():
            print(f"  CDR-H3 lengths: mean={h3_lengths[valid].mean():.1f}, "
                  f"range=[{h3_lengths[valid].min():.0f}, {h3_lengths[valid].max():.0f}]")
    return features


def _extract_all_cdrs(
    heavy_sequence: str,
    light_sequence: Optional[str],
    extract_fn,
) -> Dict[str, str]:
    """
    Extract CDR sequences for heavy (and optionally light) chain.

    Returns dict with keys: cdr_h1, cdr_h2, cdr_h3, cdr_l1, cdr_l2, cdr_l3.
    Missing CDRs are empty strings.
    """
    result = {f'cdr_h{i}': '' for i in range(1, 4)}
    result.update({f'cdr_l{i}': '' for i in range(1, 4)})

    try:
        cdrs = extract_fn(heavy_sequence, numbering='imgt', chain_type='heavy')
        if cdrs is not None:
            result['cdr_h1'] = cdrs.get('cdr1', '')
            result['cdr_h2'] = cdrs.get('cdr2', '')
            result['cdr_h3'] = cdrs.get('cdr3', '')
    except Exception:
        pass

    if light_sequence:
        try:
            cdrs_l = extract_fn(light_sequence, numbering='imgt', chain_type='light')
            if cdrs_l is not None:
                result['cdr_l1'] = cdrs_l.get('cdr1', '')
                result['cdr_l2'] = cdrs_l.get('cdr2', '')
                result['cdr_l3'] = cdrs_l.get('cdr3', '')
        except Exception:
            pass

    return result


def compute_per_cdr_physicochemical(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute per-CDR charge and hydrophobicity features.

    Per-CDR charge is strongly correlated with clearance, viscosity, and
    polyspecificity. Per-CDR hydrophobicity is the primary driver of
    aggregation propensity. (Ref: mAbs 2020, PROPERMAB 2025)

    Features (12D):
        0:  cdr_h1_charge          - Net charge of CDR-H1
        1:  cdr_h1_hydrophobicity  - Mean Kyte-Doolittle of CDR-H1
        2:  cdr_h2_charge
        3:  cdr_h2_hydrophobicity
        4:  cdr_h3_charge
        5:  cdr_h3_hydrophobicity
        6:  cdr_l1_charge          - (0 if no light chain)
        7:  cdr_l1_hydrophobicity
        8:  cdr_l2_charge
        9:  cdr_l2_hydrophobicity
        10: cdr_l3_charge
        11: cdr_l3_hydrophobicity

    Requires ANARCI for CDR extraction. Returns zeros if unavailable.

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences

    Returns:
        np.ndarray: (n_sequences, 12) per-CDR physicochemical features
    """
    n = len(heavy_sequences)
    features = np.zeros((n, 12), dtype=np.float64)

    try:
        from ..datasets.sabdab import extract_cdr_sequences
        use_anarci = True
    except (ImportError, Exception):
        use_anarci = False
        warnings.warn(
            "ANARCI not available. Per-CDR physicochemical features will be zeros. "
            "Install ANARCI: conda install -c bioconda anarci"
        )
        return features

    print(f"Computing per-CDR physicochemical features for {n} antibodies...")

    cdr_keys = ['cdr_h1', 'cdr_h2', 'cdr_h3', 'cdr_l1', 'cdr_l2', 'cdr_l3']

    for i in range(n):
        heavy = heavy_sequences[i].strip()
        light = light_sequences[i].strip() if light_sequences is not None else None

        cdrs = _extract_all_cdrs(heavy, light, extract_cdr_sequences)

        for j, key in enumerate(cdr_keys):
            cdr_seq = cdrs[key].upper()
            if not cdr_seq:
                continue
            charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0.0) for aa in cdr_seq)
            hydro = np.mean([AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0.0) for aa in cdr_seq])
            features[i, j * 2] = charge
            features[i, j * 2 + 1] = hydro

    print(f"  Generated per-CDR physicochemical features: ({n}, 12)")
    return features


def compute_extended_liability_features(
    heavy_sequences: List[str],
    light_sequences: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute extended liability motif features with CDR/framework separation.

    Extends the basic liability counts in compute_developability_features by:
    1. Splitting counts into CDR vs framework (CDR liabilities matter more)
    2. Adding DP fragmentation (defined in LIABILITY_MOTIFS but not counted)
    3. Adding Trp oxidation in CDRs (W exposed in CDRs is high risk)
    4. Computing a CDR liability concentration ratio

    Ref: LAP (PLOS Comp Biol 2024), Lu et al. (mAbs 2019)

    Features (9D):
        0: cdr_deamidation       - Deamidation motifs (NG/NS) in CDRs
        1: cdr_isomerization     - Isomerization motifs (DG/DS) in CDRs
        2: cdr_oxidation         - Met/Trp oxidation risk in CDRs
        3: cdr_glycosylation     - N-X-S/T sequons in CDRs
        4: cdr_fragmentation     - DP clipping sites in CDRs
        5: framework_deamidation - Deamidation motifs in framework
        6: framework_isomerization - Isomerization motifs in framework
        7: unpaired_cys          - Odd cysteine count (whole sequence)
        8: cdr_liability_ratio   - CDR liabilities / total liabilities

    Requires ANARCI for CDR extraction. Falls back to whole-sequence counts
    (all assigned to CDR columns, ratio=1.0) if ANARCI unavailable.

    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: Optional list of light chain sequences

    Returns:
        np.ndarray: (n_sequences, 9) extended liability features
    """
    import re

    n = len(heavy_sequences)
    features = np.zeros((n, 9), dtype=np.float64)

    try:
        from ..datasets.sabdab import extract_cdr_sequences
        use_anarci = True
    except (ImportError, Exception):
        use_anarci = False

    # High-risk motifs (subset most relevant for developability)
    deamid_motifs = ['NG', 'NS', 'NA']
    isom_motifs = ['DG', 'DS', 'DT']
    oxid_met_motifs = ['MW', 'MH', 'MD', 'MS']
    frag_motifs = ['DP']

    def count_motifs(seq, motif_list):
        return sum(seq.count(m) for m in motif_list)

    def count_trp_oxidation(seq):
        """Count exposed Trp (W) — any W in CDR is oxidation risk."""
        return seq.count('W')

    def count_glycosylation(seq):
        return len(re.findall(r'N[^P][ST]', seq))

    print(f"Computing extended liability features for {n} antibodies "
          f"(CDR/framework split={'yes' if use_anarci else 'no'})...")

    for i in range(n):
        heavy = heavy_sequences[i].upper().strip()
        light = light_sequences[i].upper().strip() if light_sequences is not None else ''
        combined = heavy + light

        if use_anarci:
            light_arg = light if light else None
            cdrs = _extract_all_cdrs(heavy, light_arg, extract_cdr_sequences)
            cdr_combined = ''.join(cdrs.values()).upper()

            # Framework = whole sequence minus CDR residues
            # Approximate: subtract CDR counts from whole-sequence counts
            fw_seq = combined  # we count whole then subtract CDR
        else:
            cdr_combined = combined
            fw_seq = ''

        # CDR liabilities
        cdr_deamid = count_motifs(cdr_combined, deamid_motifs)
        cdr_isom = count_motifs(cdr_combined, isom_motifs)
        cdr_oxid = count_motifs(cdr_combined, oxid_met_motifs) + count_trp_oxidation(cdr_combined)
        cdr_glyco = count_glycosylation(cdr_combined)
        cdr_frag = count_motifs(cdr_combined, frag_motifs)

        features[i, 0] = cdr_deamid
        features[i, 1] = cdr_isom
        features[i, 2] = cdr_oxid
        features[i, 3] = cdr_glyco
        features[i, 4] = cdr_frag

        # Framework liabilities (whole minus CDR)
        if use_anarci and fw_seq:
            whole_deamid = count_motifs(combined, deamid_motifs)
            whole_isom = count_motifs(combined, isom_motifs)
            features[i, 5] = max(0, whole_deamid - cdr_deamid)
            features[i, 6] = max(0, whole_isom - cdr_isom)

        # Unpaired cysteine (whole sequence)
        cys_count = combined.count('C')
        features[i, 7] = float(cys_count % 2 != 0)

        # CDR liability ratio
        cdr_total = cdr_deamid + cdr_isom + cdr_oxid + cdr_glyco + cdr_frag
        whole_total = (count_motifs(combined, deamid_motifs)
                       + count_motifs(combined, isom_motifs)
                       + count_motifs(combined, oxid_met_motifs) + combined.count('W')
                       + count_glycosylation(combined)
                       + count_motifs(combined, frag_motifs))
        features[i, 8] = cdr_total / whole_total if whole_total > 0 else 0.0

    print(f"  Generated extended liability features: ({n}, 9)")
    return features


# CamSol intrinsic solubility propensity values per amino acid
# From Sormanni, Aprile & Vendruscolo, J Mol Biol 427:478-490 (2015), Table 1
CAMSOL_INTRINSIC = {
    'A': -0.036, 'R':  0.602, 'N':  0.141, 'D':  0.549,
    'C': -0.245, 'E':  0.486, 'Q':  0.077, 'G': -0.198,
    'H': -0.011, 'I': -0.619, 'L': -0.538, 'K':  0.537,
    'M': -0.324, 'F': -0.770, 'P': -0.179, 'S':  0.108,
    'T': -0.021, 'W': -0.595, 'Y': -0.425, 'V': -0.463,
}


def compute_camsol_features(
    sequences: List[str],
    window_size: int = 7,
    poor_threshold: float = -1.0,
) -> np.ndarray:
    """
    Compute CamSol intrinsic solubility features.

    CamSol assigns each amino acid a solubility propensity, then applies a
    sliding window average to identify aggregation-prone patches. Distinct
    from hydrophobicity — CamSol specifically captures colloidal stability.

    Ref: Sormanni et al., J Mol Biol 427:478-490 (2015)

    Features (3D):
        0: camsol_mean         - Mean intrinsic solubility score
        1: camsol_min_window   - Worst (minimum) window score (most aggregation-prone patch)
        2: camsol_poor_frac    - Fraction of windows below poor_threshold

    Args:
        sequences: List of amino acid sequences
        window_size: Sliding window size (default 7, as in original paper)
        poor_threshold: Score below which a window is "poor" (default -1.0)

    Returns:
        np.ndarray: (n_sequences, 3) CamSol solubility features
    """
    n = len(sequences)
    features = np.zeros((n, 3), dtype=np.float64)

    print(f"Computing CamSol intrinsic solubility features for {n} sequences...")

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if not seq:
            continue

        # Per-residue intrinsic scores
        scores = np.array([CAMSOL_INTRINSIC.get(aa, 0.0) for aa in seq])

        # Feature 0: mean score
        features[i, 0] = scores.mean()

        # Sliding window averages
        if len(scores) >= window_size:
            windows = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
            features[i, 1] = windows.min()
            features[i, 2] = np.mean(windows < poor_threshold)
        else:
            features[i, 1] = scores.min()
            features[i, 2] = float(scores.mean() < poor_threshold)

    print(f"  Generated CamSol features: ({n}, 3)")
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


# ============================================================================
# PART 7: MUTATION-AWARE REPRESENTATIONS
# ============================================================================

def compute_biophysical_delta(
    mutations: List[List[Dict]],
) -> np.ndarray:
    """Compute biophysical property changes at mutation sites.

    For each sample, aggregates Δ(property) across all mutations:
    Δcharge, Δhydrophobicity, Δmw, Δvolume, Δflexibility, Δpolarity.

    Args:
        mutations: Per-sample mutation lists from FLAb loader.
            Each mutation: {'pos': int, 'wt_aa': str, 'mut_aa': str, 'chain': str}

    Returns:
        np.ndarray: (n_samples, 6) biophysical delta features.
    """
    props = ['hydrophobicity', 'charge', 'mw', 'volume', 'flexibility', 'polarity']
    n = len(mutations)
    features = np.zeros((n, 6), dtype=np.float64)

    for i, muts in enumerate(mutations):
        if not muts:
            continue
        for m in muts:
            wt_aa = m.get('wt_aa', '-')
            mut_aa = m.get('mut_aa', '-')
            if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
                for j, prop in enumerate(props):
                    features[i, j] += AA_PROPERTIES[mut_aa][prop] - AA_PROPERTIES[wt_aa][prop]

    print(f"  Generated biophysical_delta features: ({n}, 6)")
    return features


def compute_mutation_positional_features(
    mutations: List[List[Dict]],
    wt_heavy: str,
    wt_light: Optional[str] = None,
) -> np.ndarray:
    """Compute positional/structural features of mutation sites.

    Features (per sample):
    - CDR region one-hot: CDR1/CDR2/CDR3/FR1/FR2/FR3/FR4 (7D)
    - Mean BLOSUM62 substitution score at mutation sites (1D)
    - Mean relative position in sequence (1D)
    - Number of mutations (1D)
    - Max mutations in one region (1D)
    Total: 11D

    Uses simple Chothia-like CDR boundaries for heavy chain:
        CDR-H1: 26-32, CDR-H2: 52-56, CDR-H3: 95-102 (approximate)

    Args:
        mutations: Per-sample mutation lists from FLAb loader.
        wt_heavy: Wildtype heavy chain sequence.
        wt_light: Wildtype light chain sequence (optional).

    Returns:
        np.ndarray: (n_samples, 11) positional features.
    """
    # Approximate CDR boundaries (Chothia, 0-indexed)
    cdr_regions_heavy = {
        'FR1': (0, 25), 'CDR1': (26, 32), 'FR2': (33, 51),
        'CDR2': (52, 56), 'FR3': (57, 94), 'CDR3': (95, 102), 'FR4': (103, 200),
    }
    region_names = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4']

    n = len(mutations)
    features = np.zeros((n, 11), dtype=np.float64)

    seq_len = max(len(wt_heavy), 1)

    for i, muts in enumerate(mutations):
        if not muts:
            continue

        # Filter to heavy chain mutations
        heavy_muts = [m for m in muts if m.get('chain', 'heavy') == 'heavy']
        if not heavy_muts:
            features[i, 9] = len(muts)  # total mutation count
            continue

        # Region one-hot: fraction of mutations in each region
        region_counts = np.zeros(7)
        blosum_scores = []
        positions = []

        for m in heavy_muts:
            pos = m['pos']
            positions.append(pos / seq_len)

            # Determine region
            for r_idx, r_name in enumerate(region_names):
                lo, hi = cdr_regions_heavy[r_name]
                if lo <= pos <= hi:
                    region_counts[r_idx] += 1
                    break

            # BLOSUM62 score
            wt_aa = m.get('wt_aa', '-')
            mut_aa = m.get('mut_aa', '-')
            if wt_aa in BLOSUM62_MATRIX and mut_aa in BLOSUM62_MATRIX.get(wt_aa, {}):
                blosum_scores.append(BLOSUM62_MATRIX[wt_aa][mut_aa])

        total_muts = len(heavy_muts)
        # Region fractions (7D)
        features[i, :7] = region_counts / max(total_muts, 1)
        # Mean BLOSUM62 score (1D)
        features[i, 7] = np.mean(blosum_scores) if blosum_scores else 0.0
        # Mean relative position (1D)
        features[i, 8] = np.mean(positions) if positions else 0.0
        # Number of mutations (1D)
        features[i, 9] = total_muts
        # Max mutations in one region (1D)
        features[i, 10] = region_counts.max()

    print(f"  Generated positional features: ({n}, 11)")
    return features


def compute_lm_mutation_scores(
    wt_seq: str,
    mut_seqs: List[str],
    mutations: List[List[Dict]],
    model_name: str = 'ablang2',
) -> np.ndarray:
    """Compute language model mutation scores.

    For each mutation: score = log P(mut_aa | context) - log P(wt_aa | context)

    Output features (5D per sample):
    - Mean LM score across mutations (1D)
    - Min LM score (most destabilizing) (1D)
    - Max LM score (most favorable) (1D)
    - Sum of LM scores (total predicted effect) (1D)
    - Number of unfavorable mutations (score < 0) (1D)

    Falls back to BLOSUM62 proxy if LM unavailable.

    Args:
        wt_seq: Wildtype sequence.
        mut_seqs: List of mutant sequences.
        mutations: Per-sample mutation lists.
        model_name: 'ablang2' or 'esm2'.

    Returns:
        np.ndarray: (n_samples, 5) LM mutation score features.
    """
    n = len(mut_seqs)
    features = np.zeros((n, 5), dtype=np.float64)

    logits_fn = None
    aa_order_str = 'ACDEFGHIKLMNPQRSTVWY'

    if model_name == 'ablang2':
        try:
            import ablang2
            abl_model = ablang2.pretrained(model_to_use='ablang2-paired',
                                           random_init=False, device='cpu')
            abl_model.freeze()

            def _ablang2_logits(seq):
                res = abl_model([seq], ['*'], mode='likelihood')
                return res[0]  # (seq_len, vocab_size)

            logits_fn = _ablang2_logits
            print(f"  LM scoring with AbLang2...")
        except Exception as e:
            print(f"  AbLang2 failed ({e}), trying ESM-2...")
            model_name = 'esm2'

    if logits_fn is None and model_name == 'esm2':
        try:
            from transformers import AutoTokenizer, EsmForMaskedLM
            esm_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
            esm_model = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D')
            esm_model.eval()

            def _esm2_logits(seq):
                inputs = esm_tokenizer(seq, return_tensors='pt',
                                       truncation=True, max_length=512)
                with torch.no_grad():
                    output = esm_model(**inputs)
                logprobs = torch.log_softmax(output.logits[0], dim=-1)
                return logprobs[1:-1].numpy()  # strip BOS/EOS

            logits_fn = _esm2_logits
            print(f"  LM scoring with ESM-2...")
        except Exception as e:
            print(f"  ESM-2 also failed ({e}), using BLOSUM62 fallback")

    if logits_fn is None:
        # Fallback: BLOSUM62 proxy
        print(f"  Using BLOSUM62 fallback for mutation scoring")
        for i, muts in enumerate(mutations):
            if not muts:
                continue
            scores = []
            for m in muts:
                wt_aa = m.get('wt_aa', '-')
                mut_aa = m.get('mut_aa', '-')
                if wt_aa in BLOSUM62_MATRIX and mut_aa in BLOSUM62_MATRIX.get(wt_aa, {}):
                    self_score = BLOSUM62_MATRIX[wt_aa][wt_aa]
                    mut_score = BLOSUM62_MATRIX[wt_aa][mut_aa]
                    scores.append(mut_score - self_score)
            if scores:
                features[i, 0] = np.mean(scores)
                features[i, 1] = np.min(scores)
                features[i, 2] = np.max(scores)
                features[i, 3] = np.sum(scores)
                features[i, 4] = sum(1 for s in scores if s < 0)
        print(f"  Generated lm_score features (BLOSUM62 fallback): ({n}, 5)")
        return features

    # Use actual LM logits
    # Cache WT logits (compute once)
    try:
        wt_logprobs = logits_fn(wt_seq)
    except Exception as e:
        print(f"  LM logits failed for WT ({e}), falling back to zeros")
        return features

    for i, muts in enumerate(mutations):
        if not muts:
            continue
        if (i + 1) % 500 == 0:
            print(f"    LM scoring: {i+1}/{n}")

        try:
            scores = []
            for m in muts:
                if m.get('chain') != 'heavy':
                    continue
                pos = m['pos']
                if pos >= len(wt_logprobs):
                    continue
                wt_aa = m.get('wt_aa', '-')
                mut_aa = m.get('mut_aa', '-')
                if wt_aa in aa_order_str and mut_aa in aa_order_str:
                    wt_idx = aa_order_str.index(wt_aa)
                    mut_idx = aa_order_str.index(mut_aa)
                    if wt_logprobs.ndim == 2 and wt_logprobs.shape[1] >= 20:
                        score = float(wt_logprobs[pos, mut_idx] - wt_logprobs[pos, wt_idx])
                        scores.append(score)
            if scores:
                features[i, 0] = np.mean(scores)
                features[i, 1] = np.min(scores)
                features[i, 2] = np.max(scores)
                features[i, 3] = np.sum(scores)
                features[i, 4] = sum(1 for s in scores if s < 0)
        except Exception:
            pass  # Leave as zeros

    print(f"  Generated lm_score features ({model_name}): ({n}, 5)")
    return features