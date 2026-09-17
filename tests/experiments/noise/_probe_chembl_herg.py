#!/usr/bin/env python3
"""Does ChEMBL hERG hold the same compound measured in more than one paper?

n06 needs repeat measurements of one thing by different people. BindingDB has
them (3,494 drug-protein pairs measured more than once). ChEMBL might, but the
loader this repository already has averages them away before anything is
written to disk, so nobody has counted.

hERG is CHEMBL240, one of the most-measured targets in ChEMBL. This pulls that
one target, groups the records by InChIKey and counts, averaging nothing.

The number that decides it is how many compounds carry records from more than
one DOCUMENT. A compound with four records from one paper is one measurement
listed four times and two labs never disagreed about it.

    python tests/experiments/noise/_probe_chembl_herg.py

Nothing is cut on what this prints. The counts go back to the owner.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERG_TARGET = 'CHEMBL240'


CHEMBL_API = 'https://www.ebi.ac.uk/chembl/api/data'

FIELDS = ['canonical_smiles', 'standard_value', 'target_chembl_id',
          'document_chembl_id', 'assay_chembl_id', 'standard_type',
          'data_validity_comment', 'pchembl_value', 'assay_type']


def _get(url, params=None, timeout=180, attempts=6, wait=15):
    """One request, retried through ChEMBL's intermittent 500s.

    Measured on 2026-09-11: the same endpoint answered 200, then 500 ninety
    seconds later, then 200 again. A single failed call therefore says nothing
    about whether the data is reachable, and a probe that gives up on the first
    one reports "no data" when the answer is "try again".
    """
    import requests
    import time as _time

    last = None
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r, None
            last = f'HTTP {r.status_code}'
        except Exception as exc:
            last = f'{type(exc).__name__}: {exc}'
        if attempt < attempts - 1:
            print(f"    {last} — retrying in {wait}s "
                  f"({attempt + 1} of {attempts})", flush=True)
            _time.sleep(wait)
    return None, last


def api_is_up(timeout=30):
    """Whether ChEMBL's web API answers at all.

    `chembl_webresource_client` 0.10.9 builds itself from the API's /spore
    endpoint at import time and raises if that call fails, so an EBI-side
    outage looks like an import error inside the client. This asks the data
    API directly and says what came back.
    """
    r, why = _get(f'{CHEMBL_API}/status.json', timeout=timeout)
    return (r is not None), (why or 'HTTP 200')


def pull_activities(target_id, standard_type, cache_csv, limit=None,
                    page_size=1000):
    """Every activity record ChEMBL holds for one target, un-averaged.

    Goes to the REST endpoint with `requests` rather than through
    `chembl_webresource_client`, whose 0.10.9 import fails outright while the
    API's /spore endpoint is down. The filter is the same one the client would
    have sent: one target, IC50, nM, relation '='.

    Cached as a CSV so the counting can be re-run without re-querying.
    """
    if os.path.exists(cache_csv):
        cached = pd.read_csv(cache_csv)
        missing = [c for c in ('data_validity_comment', 'pchembl_value',
                               'assay_type') if c not in cached.columns]
        if not missing:
            print(f"  reading the cached raw pull: {cache_csv}")
            return cached, True
        print(f"  the cached pull at {cache_csv} has no {', '.join(missing)} "
              f"column, so it was written before these were requested. "
              f"Re-pulling.")

    up, why = api_is_up()
    if not up:
        raise RuntimeError(
            f"ChEMBL's web API is not answering ({why}). Nothing was pulled "
            f"and no count can be reported. Re-run this probe when it is "
            f"back; the counting below needs no network once "
            f"{cache_csv} exists.")

    print(f"  querying ChEMBL for {target_id} {standard_type} "
          f"(nM, relation '=')...")
    t0 = time.time()
    rows = []
    offset = 0
    while True:
        params = {
            'target_chembl_id': target_id,
            'standard_type': standard_type,
            'standard_units': 'nM',
            'standard_relation': '=',
            'format': 'json',
            'limit': page_size,
            'offset': offset,
            'only': ','.join(FIELDS),
        }
        r, why = _get(f'{CHEMBL_API}/activity', params=params)
        if r is None:
            raise RuntimeError(
                f'ChEMBL stopped answering at offset {offset}, after '
                f'{len(rows)} records: {why}')
        payload = r.json()
        page = payload.get('activities', [])
        for a in page:
            rows.append({
                'smiles_raw': a.get('canonical_smiles'),
                'standard_value': a.get('standard_value'),
                'target_chembl_id': a.get('target_chembl_id'),
                'document_chembl_id': a.get('document_chembl_id'),
                'assay_chembl_id': a.get('assay_chembl_id'),
                'standard_type': a.get('standard_type'),
                'data_validity_comment': a.get('data_validity_comment'),
                'pchembl_value': a.get('pchembl_value'),
                'assay_type': a.get('assay_type'),
            })
        total = payload.get('page_meta', {}).get('total_count')
        print(f"    {len(rows)} of {total} records "
              f"({time.time() - t0:.0f}s)", flush=True)
        if limit and len(rows) >= limit:
            break
        if not payload.get('page_meta', {}).get('next'):
            break
        offset += page_size

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df.to_csv(cache_csv, index=False)
    print(f"  {len(df)} raw records in {time.time() - t0:.0f}s, "
          f"cached to {cache_csv}")
    return df, False


def standardise(df):
    """Canonical SMILES and InChIKey, the way chembl_bioactivity does it."""
    from kirby.datasets.chembl_bioactivity import _standardize_and_inchikey
    canon, keys = _standardize_and_inchikey(df['smiles_raw'])
    df = df.copy()
    df['smiles'] = canon
    df['inchikey'] = keys
    return df


def lhasa_inchikeys():
    """Every InChIKey in Lhasa's two hERG files, or None if they are absent."""
    try:
        from experiments.federated.lhasa_herg_loader import (
            load_lhasa_herg_training, load_lhasa_herg_test,
        )
        _, _, tr_keys, _ = load_lhasa_herg_training()
        _, _, te_keys, _ = load_lhasa_herg_test()
    except Exception as exc:
        print(f"  Lhasa files unreadable ({type(exc).__name__}: {exc})")
        return None
    keys = {k for k in list(tr_keys) + list(te_keys) if k and k != 'UNKNOWN'}
    return keys


def flag_counts(df, label):
    """How many of these records carry a ChEMBL curator's validity comment.

    `data_validity_comment` is written by ChEMBL's own curation, not by this
    repository and not by any script here: it is the field that says "Potential
    author error", "Outside typical range" and so on. Every other noise number
    in the suite is scored against labels a script corrupted on purpose. This
    field is the only list of suspect records in the pipeline's data that a
    person made.

    Returns the count of flagged records, the count of molecules carrying at
    least one, and how many records each distinct comment text covers.
    """
    if 'data_validity_comment' not in df.columns:
        print(f"  {label}: no data_validity_comment column")
        return {}
    col = df['data_validity_comment']
    flagged = col.notna() & (col.astype(str).str.strip() != '') \
        & (col.astype(str).str.strip().str.lower() != 'none')
    n_flagged = int(flagged.sum())
    by_text = {str(k): int(v) for k, v in
               col[flagged].astype(str).str.strip().value_counts().items()}
    out = {'n_records': int(len(df)), 'n_records_flagged': n_flagged,
           'comment_texts': by_text}
    if 'inchikey' in df.columns:
        out['n_molecules'] = int(df['inchikey'].nunique())
        out['n_molecules_with_a_flagged_record'] = int(
            df.loc[flagged, 'inchikey'].nunique())
    print(f"  {label}: {n_flagged} records carry a curator comment, out of "
          f"{len(df)} records")
    if 'n_molecules_with_a_flagged_record' in out:
        print(f"    those sit on {out['n_molecules_with_a_flagged_record']} "
              f"molecules, out of {out['n_molecules']} molecules")
    for text, n in by_text.items():
        print(f"    {n} records: {text}")
    return out


def count_under_dataset_filter(target_id):
    """The same count under the filter the hERG dataset itself fetches with.

    `tests/alternative_data_noise_robustness.py:997` sends
    data_validity_comment__isnull=true with standard_type Ki, pchembl_value
    present and relation '=' — so the molecules ChEMBL's curators flagged are
    dropped before the dataset is written. This asks ChEMBL for the count on
    both sides of that one filter, and reads the total off page_meta rather
    than downloading the records.
    """
    base = {'target_chembl_id': target_id, 'standard_type': 'Ki',
            'pchembl_value__isnull': 'false', 'standard_relation': '=',
            'format': 'json', 'limit': 1}
    out = {}
    for name, isnull in (('kept_by_the_fetch', 'true'),
                         ('excluded_by_the_fetch', 'false')):
        params = dict(base)
        params['data_validity_comment__isnull'] = isnull
        r, why = _get(f'{CHEMBL_API}/activity', params=params)
        if r is None:
            print(f"  could not count {name}: {why}")
            return {}
        out[name] = int(r.json().get('page_meta', {}).get('total_count', 0))
    print("\n  under the hERG fetch's own filter (Ki, pchembl_value "
          "present, relation '='):")
    print(f"    {out['kept_by_the_fetch']} records have no curator comment "
          f"and are kept")
    print(f"    {out['excluded_by_the_fetch']} records carry one and are "
          f"dropped at "
          f"tests/alternative_data_noise_robustness.py:997")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', default=HERG_TARGET)
    ap.add_argument('--standard-type', default='IC50')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--output', default='tests/results/noise')
    args = ap.parse_args()

    from kirby.datasets.chembl_bioactivity import _default_data_dir
    from experiments._shared import save_experiment_results

    cache_csv = os.path.join(
        _default_data_dir('chembl'),
        f'chembl_raw_{args.target}_{args.standard_type}.csv')

    df_raw, from_cache = pull_activities(
        args.target, args.standard_type, cache_csv, limit=args.limit)
    n_raw = len(df_raw)
    print(f"  raw records: {n_raw}")

    df = standardise(df_raw)
    df['standard_value'] = pd.to_numeric(df['standard_value'],
                                         errors='coerce')
    df = df[df['standard_value'] > 0]
    df = df.dropna(subset=['smiles', 'inchikey', 'standard_value']).copy()
    n_std = len(df)
    print(f"  survived SMILES standardisation and a positive value: {n_std}")

    print()
    flags_raw = flag_counts(df_raw, 'raw pull')
    flags_std = flag_counts(df, 'after standardisation')
    flags_dataset_filter = count_under_dataset_filter(args.target)

    # pIC50, the same transform the loader uses.
    df['pActivity'] = 9 - np.log10(df['standard_value'])

    grp = df.groupby('inchikey')
    per_compound = grp.agg(
        n_records=('pActivity', 'size'),
        n_documents=('document_chembl_id', 'nunique'),
        n_assays=('assay_chembl_id', 'nunique'),
        spread=('pActivity', lambda s: float(np.std(s))),
    ).reset_index()

    n_compounds = len(per_compound)
    multi_record = per_compound[per_compound['n_records'] > 1]
    multi_doc = per_compound[per_compound['n_documents'] > 1]
    multi_assay = per_compound[per_compound['n_assays'] > 1]

    print(f"\n  compounds: {n_compounds}")
    print(f"  with more than one record:   {len(multi_record)}")
    print(f"  with more than one document: {len(multi_doc)}")
    print(f"  with more than one assay:    {len(multi_assay)}")

    def spread_stats(frame, label):
        if len(frame) == 0:
            print(f"  {label}: no compounds")
            return {'n_compounds': 0}
        sp = frame['spread'].to_numpy()
        out = {
            'n_compounds': int(len(frame)),
            'median_spread_log_units': float(np.median(sp)),
            'mean_spread_log_units': float(np.mean(sp)),
            'p90_spread_log_units': float(np.percentile(sp, 90)),
            'n_above_one_log_unit': int((sp > 1.0).sum()),
        }
        print(f"  {label}: {out['n_compounds']} compounds, spread of "
              f"p{args.standard_type[1:] if False else 'Activity'} within a "
              f"compound — median {out['median_spread_log_units']:.3f}, "
              f"mean {out['mean_spread_log_units']:.3f}, 90th percentile "
              f"{out['p90_spread_log_units']:.3f} log units; "
              f"{out['n_above_one_log_unit']} exceed 1 log unit")
        return out

    print()
    stats_multi_doc = spread_stats(multi_doc, 'across documents')
    stats_multi_record = spread_stats(multi_record, 'across records')
    # Within one assay the conditions are fixed, so this is the stricter
    # comparison: the same compound measured twice in one experiment.
    within_assay = df.groupby(['inchikey', 'assay_chembl_id']).agg(
        n_records=('pActivity', 'size'),
        spread=('pActivity', lambda s: float(np.std(s))),
    ).reset_index()
    within_assay_multi = within_assay[within_assay['n_records'] > 1]
    stats_within_assay = spread_stats(within_assay_multi,
                                      'within one assay')

    lhasa_keys = lhasa_inchikeys()
    overlap = {}
    if lhasa_keys is not None:
        keys_all = set(per_compound['inchikey'])
        keys_multi_doc = set(multi_doc['inchikey'])
        overlap = {
            'n_lhasa_molecules': len(lhasa_keys),
            'n_chembl_compounds_also_in_lhasa': len(keys_all & lhasa_keys),
            'n_multi_document_compounds_also_in_lhasa':
                len(keys_multi_doc & lhasa_keys),
        }
        print(f"\n  Lhasa holds {len(lhasa_keys)} molecules; "
              f"{len(keys_all & lhasa_keys)} of this pull's compounds are "
              f"among them, {len(keys_multi_doc & lhasa_keys)} of those "
              f"carry more than one document")

    results = {
        'target_chembl_id': args.target,
        'standard_type': args.standard_type,
        'metric': 'spread_of_pActivity_within_compound_log_units',
        'task_type': 'regression',
        'group_spread_ddof': 0,
        'raw_pull_cached_at': cache_csv,
        'read_from_cache': bool(from_cache),
        'n_raw_records': int(n_raw),
        'n_records_after_standardisation': int(n_std),
        'n_compounds': int(n_compounds),
        'n_compounds_multi_record': int(len(multi_record)),
        'n_compounds_multi_document': int(len(multi_doc)),
        'n_compounds_multi_assay': int(len(multi_assay)),
        'curator_validity_flag_raw': flags_raw,
        'curator_validity_flag_after_standardisation': flags_std,
        'curator_validity_flag_under_dataset_fetch_filter':
            flags_dataset_filter,
        'spread_across_documents': stats_multi_doc,
        'spread_across_records': stats_multi_record,
        'spread_within_one_assay': stats_within_assay,
        'lhasa_overlap': overlap,
        # Measured on the cached BindingDB file on 2026-09-11, with the SAME
        # standard deviation this probe takes of ChEMBL — the population one,
        # which is what np.std returns. An earlier note quoted 0.560, 0.983
        # and 1,148 for these three; those are the sample standard deviation
        # (ddof=1), and putting them beside a ddof=0 number would have made
        # ChEMBL look tighter than BindingDB by the choice of convention
        # alone.
        'bindingdb_for_comparison': {
            'n_groups': 3494,
            'group_spread_ddof': 0,
            'median_spread_log_units': 0.427,
            'mean_spread_log_units': 0.724,
            'n_above_one_log_unit': 909,
            'same_numbers_with_ddof_1': {'median': 0.560, 'mean': 0.983,
                                         'n_above_one_log_unit': 1148},
            'measured_on': '/Volumes/seagate/KIRBy/dta/bindingdb_kd_meta.csv',
        },
    }
    save_experiment_results(results, args.output, '_probe_chembl_herg',
                            f'{args.target}_{args.standard_type}')
    print(f"\n  written to {args.output}/_probe_chembl_herg/"
          f"{args.target}_{args.standard_type}.json")


if __name__ == '__main__':
    main()
