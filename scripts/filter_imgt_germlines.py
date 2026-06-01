#!/usr/bin/env python3
"""Filter IMGT GENE-DB combined germline FASTA by species + locus.

NCBI's IgBLAST distribution only ships pre-built germline databases for
human / mouse / rat / rabbit / rhesus_monkey. Bovine and canine need a
custom-germline setup pulled from IMGT GENE-DB. This script does exactly
that filter step.

Reads the combined IMGT reference FASTA (one file, all species, all
loci) at:

    https://www.imgt.org/download/GENE-DB/
    IMGTGENEDB-ReferenceSequences.fasta-nt-WithGaps-F+ORF+inframeP

IMGT FASTA header format (15 pipe-separated fields):

    >accession|gene_allele|species|functionality|region|...

Writes three per-segment FASTAs (V/D/J) under --out-dir with headers
reduced to just the gene_allele (so `makeblastdb -parse_seqids` works
cleanly) and IMGT-style gap dots stripped from sequences.
"""

import argparse
import re
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True,
                    help='Combined IMGT reference FASTA')
    ap.add_argument('--species', required=True,
                    help='IMGT species name e.g. "Bos taurus"')
    ap.add_argument('--species-match', choices=['exact', 'prefix'],
                    default='exact',
                    help='exact: species field must equal --species. '
                         'prefix: species field must start with --species '
                         '(needed for canine, where IMGT tags each entry '
                         'with a breed suffix like "Canis lupus familiaris_Boxer").')
    ap.add_argument('--locus', required=True, choices=['IGH', 'IGK', 'IGL'],
                    help='Receptor locus to extract')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--out-prefix', required=True,
                    help='Output filename prefix, e.g. bos_taurus_imgt')
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input FASTA missing: {args.input}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        seg: args.out_dir / f"{args.out_prefix}_{seg}.fasta"
        for seg in ('V', 'D', 'J')
    }
    out_files = {seg: p.open('w') for seg, p in out_paths.items()}
    counts = {seg: 0 for seg in ('V', 'D', 'J')}
    # Track gene_alleles already written per segment so duplicates from IMGT
    # (same allele reported from multiple accessions, or a generic "IGHD*03"
    # entry appearing more than once) don't trip `makeblastdb -parse_seqids`.
    seen = {seg: set() for seg in ('V', 'D', 'J')}
    duplicates = {seg: 0 for seg in ('V', 'D', 'J')}

    gene_re = re.compile(rf'^{re.escape(args.locus)}([VDJ])')

    current_seg = None
    current_header = None
    current_seq = []

    def flush():
        nonlocal current_seg, current_header, current_seq
        if current_seg is not None and current_header is not None:
            if current_header in seen[current_seg]:
                duplicates[current_seg] += 1
            else:
                seq = ''.join(current_seq).replace('.', '').upper()
                if seq:
                    fh = out_files[current_seg]
                    fh.write(f">{current_header}\n")
                    for i in range(0, len(seq), 60):
                        fh.write(seq[i:i + 60] + '\n')
                    counts[current_seg] += 1
                    seen[current_seg].add(current_header)
        current_seg = None
        current_header = None
        current_seq = []

    with args.input.open() as f:
        for raw in f:
            line = raw.rstrip()
            if line.startswith('>'):
                flush()
                fields = line[1:].split('|')
                if len(fields) < 3:
                    continue
                gene_allele = fields[1].strip()
                species = fields[2].strip()
                if args.species_match == 'exact':
                    if species != args.species:
                        continue
                else:
                    if not species.startswith(args.species):
                        continue
                m = gene_re.match(gene_allele)
                if not m:
                    continue
                current_seg = m.group(1)
                current_header = gene_allele
                current_seq = []
            elif current_seg is not None:
                # Drop spaces and any non-ACGTN-style characters except gap dots.
                current_seq.append(line)
    flush()

    for fh in out_files.values():
        fh.close()

    total = sum(counts.values())
    dup_total = sum(duplicates.values())
    print(f"Filtered {args.species} ({args.locus}) → {total} unique sequences"
          + (f" ({dup_total} duplicate alleles dropped)" if dup_total else "")
          + ":")
    for seg in ('V', 'D', 'J'):
        suffix = f" ({duplicates[seg]} dup)" if duplicates[seg] else ""
        print(f"  {out_paths[seg].name}: {counts[seg]}{suffix}")

    # Sanity: V should have ≥10 sequences for any real species.
    if counts['V'] < 5:
        print(f"WARNING: only {counts['V']} V-genes — check species name "
              f"spelling and locus filter.", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
