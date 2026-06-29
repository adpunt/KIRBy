"""Pre-flight stale-cache check — run BEFORE the SLURM jobs leave the queue.

For each dataset it:
  1. loads ONLY the labels (rep_names=[] -> no rep computation, no training),
  2. stats every cached rep_cache/<ds>/*.npy via mmap (reads the shape header
     only, never the full array),
  3. flags any cached rep whose row count != current label count.

A mismatch is exactly the ValueError the row-count guard would raise mid-job
(days into the queue). Here it surfaces in seconds, with the purge command.

Usage:
    python tests/preflight_cache_check.py ds1 ds2 ...      # specific datasets
    python tests/preflight_cache_check.py --all-done        # the 42 done set
    python tests/preflight_cache_check.py --all             # every DATASET_REPS
"""
import sys, os, glob
sys.path.insert(0, 'src')
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))
import numpy as np

from experiments._shared import load_data, _rep_cache_dir


def _label_count(ds):
    """Current label count with NO rep computation (rep_names=[])."""
    try:
        d = load_data(ds, rep_names=[])
        labels = d.get('labels')
        return (len(labels) if labels is not None else None), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def check(ds):
    cache_dir = _rep_cache_dir() / ds
    npys = sorted(glob.glob(str(cache_dir / '*.npy')))
    if not npys:
        return ('FRESH', ds, "no rep cache -> builds fresh, safe", [])

    n_labels, err = _label_count(ds)
    if n_labels is None:
        return ('LOADERR', ds, f"could not load labels: {err}", [])

    bad = []
    for p in npys:
        try:
            arr = np.load(p, mmap_mode='r')
            if arr.ndim >= 1 and arr.shape[0] != n_labels:
                bad.append((os.path.basename(p)[:-4], arr.shape[0]))
        except Exception as e:
            bad.append((os.path.basename(p)[:-4], f"unreadable: {e}"))
    if bad:
        return ('STALE', ds, f"{n_labels} labels but cached rows {dict(bad)}", bad)
    return ('OK', ds, f"{len(npys)} cached reps all match n={n_labels}", [])


DONE_42 = """abagym_Ang2 abagym_EGFR abagym_HER2 abagym_VEGF abagym_lysozyme
abcov_kd adme_mdr1 buchwald_hartwig caco2 ddg_combined flab_aggregation_jain
flab_aggregation_jain_hicrt flab_expression_adams flab_expression_koenig
flab_immunogenicity_marks flab_pk_jain_fcrnrt flab_pk_jain_heprt
flab_polyreact_jain_bvp flab_polyreact_jain_cic flab_thermo_jain
flab_thermo_tresanco gb_human_enhancers_cohn ginkgo_acsins ginkgo_aggregation
ginkgo_colloidal ginkgo_heparin ginkgo_hic ginkgo_pr_cho ginkgo_purity
ginkgo_titer ginkgo_tm2 herg herg_ki humanness_therapeutic nt_H3K4me3 nt_H3K9ac
nt_H3K9me3 nt_promoter_tata tcga_brca_pam50 tcga_luad_vs_lusc tz_her2_binary
tz_her2_multiclass""".split()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(2)
    if args == ['--all-done']:
        datasets = DONE_42
    elif args == ['--all']:
        from experiments.complexity.complexity_helpers import DATASET_REPS
        datasets = sorted(DATASET_REPS.keys())
    else:
        datasets = args

    results = {}
    for ds in datasets:
        status, _ds, msg, bad = check(ds)
        results.setdefault(status, []).append((ds, msg))
        tag = {'OK': '  ok ', 'FRESH': 'fresh', 'STALE': 'STALE',
               'LOADERR': 'ERR '}[status]
        print(f"[{tag}] {ds:<34} {msg}")

    print("\n" + "=" * 64)
    for s in ('STALE', 'LOADERR'):
        for ds, msg in results.get(s, []):
            if s == 'STALE':
                print(f"PURGE: python tests/purge_rep_cache.py {ds} --delete")
    n_stale = len(results.get('STALE', []))
    n_err = len(results.get('LOADERR', []))
    print(f"\nOK={len(results.get('OK', []))}  fresh={len(results.get('FRESH', []))}"
          f"  STALE={n_stale}  load-errors={n_err}")
    if n_stale == 0 and n_err == 0:
        print("ALL CLEAR — no dataset will crash on load.")
    else:
        print("ACTION NEEDED — purge the STALE datasets above before they run.")


if __name__ == '__main__':
    main()
