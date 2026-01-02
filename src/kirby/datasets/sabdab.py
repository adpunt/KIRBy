"""
SabDab (Structural Antibody Database) Dataset Loader

SabDab contains ~7000+ antibody structures with annotations including:
- Heavy/light chain sequences
- CDR definitions (IMGT/Chothia numbering)
- 3D structures (subset)
- Binding affinity data (subset)
- Paratope annotations

References:
- Dunbar et al. (2014). SAbDab: the structural antibody database. Nucleic Acids Res.
- Schneider et al. (2022). SAbDab in the age of biotherapeutics. Nucleic Acids Res.
- https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab
"""

import numpy as np
import pandas as pd
from pathlib import Path
import requests
import warnings


def download_sabdab(data_dir='data/SabDab', force=False):
    """
    Download SabDab summary data and sequences.
    
    Args:
        data_dir: Directory to store data
        force: Force re-download
    
    Returns:
        Path to downloaded data directory
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download summary
    summary_path = data_dir / 'sabdab_summary_all.tsv'
    if not summary_path.exists() or force:
        print("Downloading SabDab summary...")
        url = "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(summary_path, 'wb') as f:
                f.write(response.content)
            print(f"  Downloaded summary to {summary_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download SabDab summary: {e}")
    else:
        print(f"SabDab summary already exists at {summary_path}")
    
    # Download sequences file
    sequences_path = data_dir / 'sabdab_all.fasta'
    if not sequences_path.exists() or force:
        print("Downloading SabDab sequences (this may take several minutes)...")
        # SabDab provides a FASTA download of all sequences
        url = "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?all=true"
        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            
            # Parse the HTML to get the actual FASTA download link
            # Or use their direct FASTA endpoint
            fasta_url = "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/dumseq/"
            response = requests.post(
                fasta_url,
                data={'all': 'true'},
                timeout=300
            )
            response.raise_for_status()
            
            with open(sequences_path, 'wb') as f:
                f.write(response.content)
            print(f"  Downloaded sequences to {sequences_path}")
        except Exception as e:
            print(f"  WARNING: Could not download sequences: {e}")
            print(f"  You can manually download from: http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/")
            sequences_path = None
    else:
        print(f"SabDab sequences already exist at {sequences_path}")
    
    return data_dir


def fetch_sequences_from_sabdab(pdb_ids, chain_info):
    """
    Fetch antibody sequences directly from SabDab using their API endpoints.
    
    Based on the official SabDab download script.
    URL pattern: https://opig.stats.ox.ac.uk/webapps/abdb/entries/{pdb}/sequences/{pdb}_{chain}_VH.fa
    
    Args:
        pdb_ids: List of PDB IDs
        chain_info: Dict mapping PDB IDs to chain identifiers
    
    Returns:
        dict: {pdb_id: {'heavy': seq, 'light': seq}}
    """
    import urllib.request
    
    print(f"Fetching sequences from SabDab for {len(pdb_ids)} structures...")
    
    sequences = {}
    
    for i, pdb_id in enumerate(pdb_ids):
        if (i + 1) % 10 == 0:
            print(f"  Fetched {i+1}/{len(pdb_ids)}...")
        
        # Get chain identifiers
        if pdb_id not in chain_info:
            continue
        
        heavy_chain = chain_info[pdb_id].get('heavy_chain')
        light_chain = chain_info[pdb_id].get('light_chain')
        
        if not heavy_chain or not light_chain or heavy_chain == 'NA' or light_chain == 'NA':
            continue
        
        sequences[pdb_id] = {}
        
        try:
            # Fetch heavy chain VH sequence
            heavy_url = f"https://opig.stats.ox.ac.uk/webapps/abdb/entries/{pdb_id}/sequences/{pdb_id}_{heavy_chain}_VH.fa"
            with urllib.request.urlopen(heavy_url, timeout=10) as response:
                fasta_content = response.read().decode('utf-8')
                # Parse FASTA (skip header line, join remaining lines, remove whitespace)
                lines = fasta_content.strip().split('\n')
                if len(lines) > 1:
                    # Join all non-header lines and remove any whitespace/newlines
                    seq = ''.join(lines[1:]).replace('\n', '').replace('\r', '').replace(' ', '').strip()
                    sequences[pdb_id]['heavy'] = seq
            
            # Fetch light chain VL sequence
            light_url = f"https://opig.stats.ox.ac.uk/webapps/abdb/entries/{pdb_id}/sequences/{pdb_id}_{light_chain}_VL.fa"
            with urllib.request.urlopen(light_url, timeout=10) as response:
                fasta_content = response.read().decode('utf-8')
                lines = fasta_content.strip().split('\n')
                if len(lines) > 1:
                    seq = ''.join(lines[1:]).replace('\n', '').replace('\r', '').replace(' ', '').strip()
                    sequences[pdb_id]['light'] = seq
            
            # Verify both chains fetched
            if 'heavy' not in sequences[pdb_id] or 'light' not in sequences[pdb_id]:
                del sequences[pdb_id]
        
        except Exception as e:
            if i < 5:  # Show first few errors
                print(f"  Warning: Could not fetch {pdb_id}: {e}")
            if pdb_id in sequences:
                del sequences[pdb_id]
            continue
    
    print(f"Successfully fetched {len(sequences)} complete antibody sequences")
    return sequences


def parse_sabdab_fasta(fasta_path):
    """
    Parse SabDab FASTA file to extract sequences.
    
    Returns:
        dict: {pdb_id: {'heavy': seq, 'light': seq, ...}}
    """
    try:
        from Bio import SeqIO
    except ImportError:
        raise ImportError("Sequence parsing requires BioPython: pip install biopython")
    
    sequences = {}
    
    with open(fasta_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            # Parse header: >PDB_ID|CHAIN|TYPE
            # Example: >1A2Y|H|VH or >1A2Y|L|VL
            parts = record.id.split('|')
            if len(parts) >= 3:
                pdb_id = parts[0]
                chain = parts[1]
                chain_type = parts[2]
                
                if pdb_id not in sequences:
                    sequences[pdb_id] = {}
                
                # Store by chain type
                if 'VH' in chain_type or chain_type == 'H':
                    sequences[pdb_id]['heavy'] = str(record.seq)
                elif 'VL' in chain_type or chain_type == 'L' or 'VK' in chain_type:
                    sequences[pdb_id]['light'] = str(record.seq)
    
    print(f"Parsed {len(sequences)} structures with sequences")
    return sequences


def parse_sabdab_summary(csv_path):
    """Parse SabDab summary TSV file"""
    
    df = pd.read_csv(csv_path, sep='\t', low_memory=False)
    
    print(f"Loaded {len(df)} antibody structures")
    print(f"Columns: {list(df.columns)}")
    
    # Key columns:
    # - pdb: PDB ID
    # - Hchain/Lchain: chain IDs
    # - antigen_chain: if bound to antigen
    # - resolution: structure resolution
    # - method: X-ray, EM, etc.
    # - scfv: single-chain variable fragment
    # - engineered: Yes/No
    # - heavy_species/light_species
    
    return df


def load_sabdab_affinity(data_dir='data/SabDab', source='skempi'):
    """
    Load SabDab sequences with binding affinity data.
    
    Since SabDab doesn't include affinity measurements directly, this function
    integrates with external databases:
    - SKEMPI 2.0: Protein-protein interaction database with ΔΔG values
    - Future: AB-Bind for antibody-specific mutations
    
    Args:
        data_dir: Data directory
        source: 'skempi' or 'abbind'
    
    Returns:
        dict: {
            'heavy_seqs': [...],
            'light_seqs': [...], 
            'affinity': [...],  # ΔΔG values in kcal/mol
            'pdb_ids': [...],
            'mutations': [...],  # Optional mutation information
            'metadata': {...}
        }
    """
    if source == 'skempi':
        return _load_skempi_affinity(data_dir)
    else:
        raise ValueError(f"Unknown affinity source: {source}")


def _load_skempi_affinity(data_dir='data/SabDab'):
    """
    Load affinity data from SKEMPI 2.0 database.
    
    SKEMPI 2.0: https://life.bsc.es/pid/skempi2
    Contains mutation effects on protein-protein interactions including antibodies.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    skempi_file = data_dir / 'skempi_v2.csv'
    
    # Download SKEMPI 2.0
    if not skempi_file.exists():
        print("Downloading SKEMPI 2.0 database...")
        url = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(skempi_file, 'wb') as f:
                f.write(response.content)
            
            print(f"  Downloaded to {skempi_file}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download SKEMPI 2.0: {e}\n"
                f"Please download manually from: https://life.bsc.es/pid/skempi2/database/download"
            )
    else:
        print(f"SKEMPI 2.0 already downloaded at {skempi_file}")
    
    # Parse SKEMPI
    print("Parsing SKEMPI 2.0...")
    df = pd.read_csv(skempi_file, delimiter=';')
    
    print(f"Loaded {len(df)} mutations from SKEMPI 2.0")
    
    # Filter for antibody-antigen complexes
    # SKEMPI has a 'Protein 1' and 'Protein 2' column
    # Look for entries with antibody indicators
    antibody_keywords = ['antibody', 'fab', 'fv', 'immunoglobulin', 'nanobody']
    
    def is_antibody_complex(row):
        p1 = str(row.get('Protein 1', '')).lower()
        p2 = str(row.get('Protein 2', '')).lower()
        return any(kw in p1 or kw in p2 for kw in antibody_keywords)
    
    ab_df = df[df.apply(is_antibody_complex, axis=1)]
    
    print(f"Found {len(ab_df)} antibody-related entries")
    
    if len(ab_df) == 0:
        raise RuntimeError("No antibody complexes found in SKEMPI 2.0")
    
    # Extract PDB IDs and affinity changes
    pdb_ids = ab_df['#Pdb'].str.strip().tolist()
    ddg_values = ab_df['Affinity_mut_parsed'].values  # ΔΔG in kcal/mol
    mutations = ab_df['Mutation(s)_cleaned'].tolist()
    
    # Now we need sequences - fetch from SabDab using PDB IDs
    print("Fetching sequences from SabDab API...")
    
    # Download summary to get chain info
    sabdab_dir = download_sabdab(data_dir)
    summary_path = sabdab_dir / 'sabdab_summary_all.tsv'
    sabdab_df = pd.read_csv(summary_path, sep='\t', low_memory=False)
    
    # Build chain info for SKEMPI PDBs
    chain_info = {}
    for pdb_id in pdb_ids:
        pdb_lower = pdb_id.lower().strip()
        matching = sabdab_df[sabdab_df['pdb'].str.lower() == pdb_lower]
        if len(matching) > 0:
            row = matching.iloc[0]
            chain_info[pdb_lower] = {
                'heavy_chain': row.get('Hchain'),
                'light_chain': row.get('Lchain')
            }
    
    # Fetch sequences using SabDab API
    all_sequences = fetch_sequences_from_sabdab(list(chain_info.keys()), chain_info)
    
    # Match sequences to SKEMPI entries
    heavy_seqs = []
    light_seqs = []
    affinity_values = []
    valid_pdb_ids = []
    valid_mutations = []
    
    for pdb_id, ddg, mut in zip(pdb_ids, ddg_values, mutations):
        pdb_id_clean = pdb_id.lower().strip()
        
        if pdb_id_clean in all_sequences:
            seqs = all_sequences[pdb_id_clean]
            
            if 'heavy' in seqs and 'light' in seqs:
                heavy_seqs.append(seqs['heavy'])
                light_seqs.append(seqs['light'])
                affinity_values.append(ddg)
                valid_pdb_ids.append(pdb_id)
                valid_mutations.append(mut)
    
    print(f"Matched {len(heavy_seqs)} complexes with sequences")
    
    if len(heavy_seqs) == 0:
        raise RuntimeError(
            "Could not match SKEMPI antibody entries with SabDab sequences. "
            "This may be due to PDB ID formatting differences."
        )
    
    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'affinity': np.array(affinity_values),  # ΔΔG in kcal/mol
        'pdb_ids': valid_pdb_ids,
        'mutations': valid_mutations,
        'metadata': {
            'source': 'SKEMPI 2.0',
            'n_complexes': len(heavy_seqs),
            'affinity_type': 'ΔΔG (kcal/mol)',
            'description': 'Mutation effects on antibody-antigen binding'
        }
    }


def load_sabdab_sequences(
    data_dir='data/SabDab',
    task='structure',
    n_samples=1000,
    paired_only=True,
    resolution_cutoff=3.0,
    remove_redundancy=False,
    random_seed=42
):
    """
    Load SabDab antibody sequences for various tasks.
    
    Args:
        data_dir: Data directory
        task: 'structure', 'paratope', 'humanness', 'developability'
        n_samples: Number of samples to load
        paired_only: Only include paired heavy+light chains
        resolution_cutoff: Maximum resolution (Å)
        remove_redundancy: Remove >95% sequence identity
        random_seed: Random seed
    
    Returns:
        dict: {
            'heavy_seqs': [...],
            'light_seqs': [...],
            'labels': [...],  # task-dependent
            'pdb_ids': [...],
            'metadata': {...}
        }
    """
    # Download data
    data_dir = download_sabdab(data_dir)
    
    # Load summary
    summary_path = data_dir / 'sabdab_summary_all.tsv'
    df = pd.read_csv(summary_path, sep='\t', low_memory=False)
    
    print(f"\nLoaded {len(df)} structures from SabDab")
    print(f"Loading sequences for task: {task}")
    
    # Filter criteria
    filters = []
    
    # Paired chains only
    if paired_only:
        before = len(df)
        df = df[(df['Hchain'].notna()) & (df['Lchain'].notna())]
        df = df[df['scfv'] != 'True']  # Exclude single-chain
        print(f"  After paired filter: {before} → {len(df)}")
        filters.append("paired heavy+light")
    
    # Resolution
    if resolution_cutoff:
        before = len(df)
        df = df[df['resolution'].notna()]
        df['resolution'] = pd.to_numeric(df['resolution'], errors='coerce')
        df = df[df['resolution'] <= resolution_cutoff]
        print(f"  After resolution ≤{resolution_cutoff}Å filter: {before} → {len(df)}")
        filters.append(f"resolution ≤ {resolution_cutoff}Å")
    
    # Method (prefer X-ray) - make this optional
    if 'method' in df.columns:
        before = len(df)
        # Don't filter, just prioritize X-ray if available
        print(f"  Methods available: {df['method'].value_counts().head()}")
        # df = df[df['method'].str.contains('X-ray|diffraction', case=False, na=False)]
        # print(f"  After X-ray filter: {before} → {len(df)}")
        # filters.append("X-ray structures")
    
    print(f"Filters applied: {', '.join(filters)}")
    print(f"Remaining: {len(df)} structures")
    
    # Sample
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_seed)
    
    print(f"Selected: {len(df)} structures")
    
    # Load sequences
    sequences_path = data_dir / 'sabdab_all.fasta'
    
    if sequences_path.exists():
        # Parse FASTA if available
        print("Using cached FASTA sequences...")
        all_sequences = parse_sabdab_fasta(sequences_path)
    else:
        # Fallback: Fetch from SabDab directly (this is normal!)
        print("Fetching sequences directly from SabDab API (this is normal)...")
        
        # Build chain info dict for the sampled structures only
        chain_info = {}
        for _, row in df.iterrows():
            pdb_id = row['pdb']
            chain_info[pdb_id] = {
                'heavy_chain': row.get('Hchain'),
                'light_chain': row.get('Lchain')
            }
        
        # Fetch sequences for these specific PDBs
        all_sequences = fetch_sequences_from_sabdab(df['pdb'].tolist(), chain_info)

    
    # Match sequences to filtered PDBs
    heavy_seqs = []
    light_seqs = []
    pdb_ids = []
    
    for pdb_id in df['pdb'].values:
        if pdb_id in all_sequences:
            seqs = all_sequences[pdb_id]
            if 'heavy' in seqs and 'light' in seqs:
                heavy_seqs.append(seqs['heavy'])
                light_seqs.append(seqs['light'])
                pdb_ids.append(pdb_id)
    
    print(f"Found sequences for {len(heavy_seqs)}/{len(df)} structures")
    
    if len(heavy_seqs) == 0:
        raise RuntimeError("No sequences found for selected structures")
    
    # For structure prediction task, labels could be resolution
    if task == 'structure':
        labels = df[df['pdb'].isin(pdb_ids)]['resolution'].values
    else:
        # For other tasks, we don't have labels yet
        labels = np.zeros(len(heavy_seqs))
    
    data = {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'pdb_ids': pdb_ids,
        'metadata': {
            'task': task,
            'n_structures': len(heavy_seqs),
            'filters': filters
        }
    }
    
    return data


def extract_cdr_sequences(sequence, numbering='imgt', chain_type='heavy'):
    """
    Extract CDR sequences using ANARCI numbering.
    
    Args:
        sequence: Antibody sequence string
        numbering: 'imgt', 'chothia', 'kabat'
        chain_type: 'heavy' or 'light'
    
    Returns:
        dict: {
            'cdr1': str,
            'cdr2': str,
            'cdr3': str,
            'framework1': str,
            'framework2': str,
            'framework3': str,
            'framework4': str,
            'numbered_seq': [(position, residue), ...]
        }
    """
    try:
        from anarci import anarci
    except ImportError:
        raise ImportError(
            "CDR extraction requires ANARCI: conda install -c bioconda anarci"
        )
    
    # Clean sequence - ANARCI only accepts standard amino acids
    # Remove ANY non-standard characters including numbers, newlines, spaces, etc.
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    cleaned_sequence = ''.join([aa.upper() for aa in sequence if aa.upper() in valid_aa])
    
    if len(cleaned_sequence) == 0:
        warnings.warn(f"Sequence contains no valid amino acids: {sequence[:20]}...")
        return None
    
    if len(cleaned_sequence) < len(sequence):
        removed = len(sequence) - len(cleaned_sequence)
        if removed > 50:  # Only warn if >50 characters removed
            warnings.warn(
                f"Removed {removed} non-standard characters from sequence (length {len(sequence)} → {len(cleaned_sequence)})"
            )
    
    # Run ANARCI
    results = anarci([('seq', cleaned_sequence)], scheme=numbering, output=False)
    
    if results is None or len(results[0]) == 0:
        warnings.warn(f"Could not number sequence: {cleaned_sequence[:20]}...")
        return None
    
    numbering_result, chain_type_result, species = results[0][0]
    
    # ANARCI numbering format - handle flexibly
    numbered_seq = []
    for item in numbering_result:
        # Try to unpack - ANARCI format varies
        try:
            if len(item) == 2:
                # Could be ((pos, ins), aa) or (pos, aa)
                first, second = item
                if isinstance(first, tuple) and len(first) == 2:
                    # Format: ((pos, ins), aa)
                    pos, ins = first
                    aa = second
                else:
                    # Format: (pos, aa) where pos might be tuple or int
                    if isinstance(first, tuple):
                        pos, ins = first
                        aa = second
                    else:
                        pos = first
                        ins = ''
                        aa = second
            else:
                # Unknown format - skip
                continue
            
            # Skip gaps
            if aa == '-':
                continue
            
            # Store with position key
            if ins and ins != ' ':
                numbered_seq.append((f"{pos}{ins}", aa))
            else:
                numbered_seq.append((pos, aa))
                
        except (ValueError, TypeError) as e:
            # Can't unpack this item - skip it
            continue
    
    # Define CDR regions (IMGT numbering)
    if numbering == 'imgt':
        if chain_type == 'heavy':
            cdr_ranges = {
                'cdr1': (27, 38),
                'cdr2': (56, 65),
                'cdr3': (105, 117)
            }
            fw_ranges = {
                'framework1': (1, 26),
                'framework2': (39, 55),
                'framework3': (66, 104),
                'framework4': (118, 128)
            }
        else:  # light
            cdr_ranges = {
                'cdr1': (27, 38),
                'cdr2': (56, 65),
                'cdr3': (105, 117)
            }
            fw_ranges = {
                'framework1': (1, 26),
                'framework2': (39, 55),
                'framework3': (66, 104),
                'framework4': (118, 127)
            }
    else:
        raise NotImplementedError(f"CDR ranges not defined for {numbering}")
    
    # Extract sequences
    pos_dict = {}
    for pos, aa in numbered_seq:
        # Handle both integer positions and string positions (with insertion codes)
        if isinstance(pos, str):
            # Extract base position from strings like "82A"
            base_pos = int(''.join(c for c in pos if c.isdigit()))
            pos_dict[base_pos] = pos_dict.get(base_pos, '') + aa
        else:
            pos_dict[pos] = aa
    
    def get_region(start, end):
        seq = []
        for i in range(start, end + 1):
            if i in pos_dict:
                seq.append(pos_dict[i])
        return ''.join(seq)
    
    cdrs = {name: get_region(*rng) for name, rng in cdr_ranges.items()}
    frameworks = {name: get_region(*rng) for name, rng in fw_ranges.items()}
    
    return {
        **cdrs,
        **frameworks,
        'numbered_seq': numbered_seq,
        'full_seq': sequence
    }


def get_imgt_positions(numbered_seq, positions):
    """
    Extract specific IMGT positions from numbered sequence.
    
    Args:
        numbered_seq: List of (position, residue) tuples
        positions: List of IMGT positions to extract
    
    Returns:
        dict: {position: residue}
    """
    pos_dict = {pos: aa for pos, aa in numbered_seq}
    return {pos: pos_dict.get(pos, '-') for pos in positions}


# Key IMGT positions for antibody binding (from literature)
IMGT_KEY_POSITIONS = {
    'heavy': {
        'cdr_h1': list(range(27, 39)),
        'cdr_h2': list(range(56, 66)),
        'cdr_h3': list(range(105, 118)),  # Most important!
        'framework_h': [1, 5, 9, 11, 23, 41, 49, 94]  # Key framework positions
    },
    'light': {
        'cdr_l1': list(range(27, 39)),
        'cdr_l2': list(range(56, 66)),
        'cdr_l3': list(range(105, 118)),
        'framework_l': [2, 4, 35, 38, 46, 87, 98]
    }
}


# Metadata for reference
SABDAB_METADATA = {
    'name': 'SabDab',
    'full_name': 'Structural Antibody Database',
    'n_structures': '7000+',
    'url': 'https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab',
    'reference': 'Dunbar et al. (2014). Nucleic Acids Res. 42:D1140-D1146',
    'tasks': [
        'Structure prediction (CDR loops)',
        'Paratope prediction',
        'Binding affinity (with external data)',
        'Antibody-antigen docking',
        'Humanness/developability'
    ],
    'key_finding': 'CDR-H3 is most variable and important for binding',
    'benchmarks': {
        'IgFold': 'CDR-H3 RMSD ~2.8Å',
        'ImmuneBuilder': 'CDR-H3 RMSD ~2.81Å', 
        'Paragraph': 'Paratope F1 ~0.68'
    }
}


def get_sabdab_info():
    """Print information about SabDab dataset"""
    info = SABDAB_METADATA
    
    print("="*70)
    print(f"Dataset: {info['full_name']} ({info['name']})")
    print("="*70)
    print(f"Structures:  {info['n_structures']}")
    print(f"URL:         {info['url']}")
    print(f"\nCommon Tasks:")
    for task in info['tasks']:
        print(f"  - {task}")
    print(f"\nKey Insight: {info['key_finding']}")
    print(f"\nSOTA Benchmarks:")
    for model, perf in info['benchmarks'].items():
        print(f"  {model}: {perf}")
    print(f"\nReference: {info['reference']}")
    print("="*70)