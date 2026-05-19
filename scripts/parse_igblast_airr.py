#!/usr/bin/env python3
"""Parse IgBLAST AIRR-format TSV output into a deduplicated VH FASTA.

Called by tests/slurm_scripts/submit_bovine_assembly.sh as step 6 of the
T-Ab.2 bovine VH assembly pipeline. Reads every *.airr.tsv under
--airr-dir, filters productive == T + locus == IGH, extracts the
`sequence_alignment_aa` column (the IMGT-gapped translation of the VH
region), strips the IMGT gaps to recover ungapped amino acids, dedups by
that canonical sequence, and writes a FASTA + a small JSON report.

AIRR-rearrangement schema reference:
    https://docs.airr-community.org/en/stable/datarep/rearrangements.html
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


AIRR_REQUIRED = (
    'sequence_id', 'productive', 'locus', 'sequence_alignment_aa', 'v_call',
)


def _canonicalise(seq: str) -> str:
    """Strip IMGT gaps + uppercase to give one canonical AA form per VH."""
    if not seq:
        return ''
    return seq.replace('.', '').replace('-', '').upper()


def _iter_airr_rows(path: Path) -> Iterable[Dict[str, str]]:
    """Yield AIRR rows from one IgBLAST TSV file (skips malformed rows)."""
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if all(k in row for k in AIRR_REQUIRED):
                yield row


def _is_productive_igh(row: Dict[str, str]) -> bool:
    """AIRR 'productive' = T / TRUE / true and locus = IGH."""
    if (row.get('locus') or '').upper() != 'IGH':
        return False
    prod = (row.get('productive') or '').strip().upper()
    return prod in ('T', 'TRUE', '1', 'YES')


def parse_airr_dir(
    airr_dir: Path,
    min_vh_length: int = 90,
    max_vh_length: int = 250,
) -> Dict[str, object]:
    """Walk every *.airr.tsv in airr_dir, return dedup'd VH list + counts.

    Returns a dict with:
        unique_vh: List[str]   — deduplicated canonical AA sequences
        per_sample_raw: Dict[str, int]   — total rows read per file
        per_sample_productive: Dict[str, int]
        per_sample_dedup_added: Dict[str, int]
        dropped_short: int
        dropped_long: int
        first_v_gene: Dict[str, str]   — canonical AA → first observed V gene call
    """
    seen: set = set()
    unique_vh: List[str] = []
    first_v_gene: Dict[str, str] = {}
    per_sample_raw: Counter = Counter()
    per_sample_productive: Counter = Counter()
    per_sample_dedup_added: Counter = Counter()
    dropped_short = 0
    dropped_long = 0

    files = sorted(airr_dir.glob('*.airr.tsv'))
    if not files:
        raise RuntimeError(f"No *.airr.tsv files in {airr_dir}")

    for path in files:
        sample = path.stem.replace('.airr', '')
        for row in _iter_airr_rows(path):
            per_sample_raw[sample] += 1
            if not _is_productive_igh(row):
                continue
            per_sample_productive[sample] += 1
            aa = _canonicalise(row.get('sequence_alignment_aa') or '')
            if len(aa) < min_vh_length:
                dropped_short += 1
                continue
            if len(aa) > max_vh_length:
                dropped_long += 1
                continue
            if aa in seen:
                continue
            seen.add(aa)
            unique_vh.append(aa)
            per_sample_dedup_added[sample] += 1
            first_v_gene[aa] = row.get('v_call', '') or ''

    return {
        'unique_vh': unique_vh,
        'per_sample_raw': dict(per_sample_raw),
        'per_sample_productive': dict(per_sample_productive),
        'per_sample_dedup_added': dict(per_sample_dedup_added),
        'dropped_short': dropped_short,
        'dropped_long': dropped_long,
        'first_v_gene': first_v_gene,
        'min_vh_length': min_vh_length,
        'max_vh_length': max_vh_length,
    }


def write_fasta(unique_vh: List[str],
                first_v_gene: Dict[str, str],
                out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for i, aa in enumerate(unique_vh):
            vgene = first_v_gene.get(aa, '')
            header = f">bovine_vh_{i:06d} v_call={vgene}"
            f.write(header + '\n')
            # 60-char wrap for FASTA compatibility
            for j in range(0, len(aa), 60):
                f.write(aa[j:j + 60] + '\n')


def write_report(stats: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop the heavy fields the FASTA already represents.
    summary = {k: v for k, v in stats.items()
               if k not in ('unique_vh', 'first_v_gene')}
    summary['n_unique_vh'] = len(stats['unique_vh'])  # type: ignore[arg-type]
    out_path.write_text(json.dumps(summary, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--airr-dir', type=Path, required=True,
                    help='Directory of *.airr.tsv files from igblastn -outfmt 19')
    ap.add_argument('--out-fasta', type=Path, required=True)
    ap.add_argument('--out-report', type=Path, required=True)
    ap.add_argument('--min-vh-length', type=int, default=90,
                    help='Drop AA sequences shorter than this (default 90).')
    ap.add_argument('--max-vh-length', type=int, default=250,
                    help='Drop AA sequences longer than this (default 250).')
    args = ap.parse_args(argv)

    stats = parse_airr_dir(args.airr_dir,
                           min_vh_length=args.min_vh_length,
                           max_vh_length=args.max_vh_length)
    write_fasta(stats['unique_vh'], stats['first_v_gene'], args.out_fasta)  # type: ignore[arg-type]
    write_report(stats, args.out_report)
    print(f"  wrote {len(stats['unique_vh'])} unique VH to {args.out_fasta}")  # type: ignore[arg-type]
    print(f"  report: {args.out_report}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
