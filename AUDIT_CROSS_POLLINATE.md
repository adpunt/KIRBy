# Cross-Pollination Audit (D.9 — ANNITIA)

**Date:** 2026-05-08
**Scope:** Verify exhaustiveness of D.9 endpoint cross-pollination in `tests/experiments/annitia/cross_pollinate.py` (BE-stage donors -> CD-stage recipients).

## TL;DR

The original `experiment_cd.py` only ran **ONE** D9 cross-poll cell per endpoint — using
the best CD-stage Cox/RSF baseline as donor and `SurvivalCoxPH` as recipient
(`D9-crosspoll-cox`, hepatic C=0.6807, death C=0.8091).

The 96 **BE-stage OOFs** (8 algos x 6 view-sets x 2 endpoints) were never used as donors.
This audit ran the missing combinations on Mac CPU using the same recipient
pipeline (recipient view = `snapshot+clinical_pro+fpca`, recipient model = Cox).

## Recipe

For each direction (donor_endpoint -> recipient_endpoint):
1. Donor OOF = `tests/results/annitia/be/oof_<donor_ep>_<viewset>__<algo>.npy`.
2. Recipient features = `snapshot+clinical_pro+fpca` (130 dims), same as `D9-crosspoll-cox`.
3. Recipient model = `CrossPollinateSurvival(SurvivalCoxPH, donor_oof_path=...)`.
4. 3-fold CV on training -> OOF; refit on full valid -> val_test.
5. Saved as `oof_<recipient>_D9x__<viewset>__<algo>.npy` and `val_<recipient>_D9x__<viewset>__<algo>.npy`.
6. Updated `tests/results/annitia/cd/cv_summary.json`.

## Coverage matrix

Two directions x 6 view-sets x 8 algos = **96 target cells**.

(Filled in below after the runner completes.)

## Auto-inclusion in meta_avg pool

`tests/experiments/annitia/refuse_meta_aware.py` and `refuse_variants.py` walk
`pool_dirs = [be, cd, hep_boost, branches, external]` and call `_collect(d, ep)`
which iterates **every** `per_condition[ep]` entry whose `oof_<ep>_<cid>.npy` and
`val_<ep>_<cid>.npy` files exist on disk. Concretely (`refuse_variants.py:111-129`):

    for r in sm.get('per_condition', {}).get(endpoint, []):
        oof_p = d / f'oof_{endpoint}_{cid}.npy'
        val_p = d / f'val_{endpoint}_{cid}.npy'
        if not (oof_p.exists() and val_p.exists()): continue
        out.append(...)

So **every new `D9x__*` record in `cd/cv_summary.json` is auto-included** — no edits
to `refuse_meta_aware.py` are needed.

## Top-5 cross-pollinated outputs (by C-index)

(Filled in below after the runner completes.)

## Notes on what was added

- 96 new `oof_*_D9x__*.npy` files (or fewer if process timed out before death donors finished).
- 96 new `val_*_D9x__*.npy` files (matching).
- `tests/results/annitia/cd/cv_summary.json` updated with `D9x__*` entries.

The original `D9-crosspoll-cox` entry remains untouched.
