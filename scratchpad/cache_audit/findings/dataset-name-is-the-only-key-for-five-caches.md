# dataset-name-is-the-only-key-for-five-caches

**title:** A dataset rebuilt under the same name silently invalidates five row-indexed caches, and no procedure enforces a wipe

**area:** tests/experiments/similarity_metrics/ — every .py file in the directory

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/similarity_metrics/build_local_datasets.py:48-51; build_adme_solubility.py:50-52; build_moleculeace.py:113-115; consumers keyed only by name at _data.py:55, shape_similarity.py:56-89, aug3d_similarity.py:66-107, augmented_descriptor_reps.py:132-145, learned_embeddings.py:128-134, mcs_cache.py:72

## scenario as reported
Run 1 builds sim_herg_ki.npz, then mcs_pairs_herg_ki.npz (8000 pairs), sim_herg_ki_spatial.npz (1415 molecules of usr/usrcat/pharm3d), sim_herg_ki_3daug.npz (1415 of e3fp/coulomb) and sim_herg_ki_augmented.npz (1415 of pdv/mordred). Run 2 is `python build_local_datasets.py herg_ki` after the source CSV changed — line 49 does np.savez_compressed over the same path with no comparison to what was there. The four derived files are untouched and still hold 1415 rows in the OLD molecule order. shape_similarity.py:74-76 then reports `spatial cache hit (built 1415 >= need 1415)` and hands back the old conformer descriptors as this dataset's 3D block; creative_hybrids_v2.load_pharm3d_full (creative_hybrids_v2.py:98-112) does the same and similarity_noise_cleaning.py:141 scores its shape combinations on it. Row 900's shape descriptor now belongs to a different molecule than row 900's label. Nothing compares anything: the only thing checked is n_built >= n.

## evidence as reported
build_local_datasets.py:48-51 writes unconditionally. shape_similarity.py:71 `have_n = int(cached['n_built']) ... else 0` and :74 `if have_all and have_n >= need_n:` is the entire validation. aug3d_similarity.py:83-90 is the same two lines. augmented_descriptor_reps.py:194 `if rep in cached and cached[rep].shape[0] >= n:` is the same check by row count. learned_embeddings.py:129 is `if os.path.exists(out): print('cached'); continue`. On disk the derived files are dated 19 July while the JSON results beside them are dated 3-4 September. No script in scripts/ or tests/ deletes these paths — they sit inside the repo source tree at tests/experiments/similarity_metrics/, not under $KIRBY_CACHE_DIR, so wiping ~/.cache/kirby leaves every one of them in place.

## fix as proposed
Two parts, and the second is the one that needs your decision. First: make each builder record the identity of what it wrote — a hash of the SMILES list in the same npz — and make _data.load_prepared return it, so each derived cache can store it and refuse a mismatch. Second: until that exists the procedure is manual and undocumented, and it is `rm tests/experiments/similarity_metrics/{mcs_pairs,sim}_<name>*.npz` plus the sim_data/emb_<name>_*.npy files, immediately after any run of build_local_datasets.py, build_adme_solubility.py or build_moleculeace.py. Rebuilding costs roughly 12 minutes of MCS pairs plus the conformer passes per dataset. It is worth deciding whether these five caches should move under $KIRBY_CACHE_DIR so one wipe reaches them.
