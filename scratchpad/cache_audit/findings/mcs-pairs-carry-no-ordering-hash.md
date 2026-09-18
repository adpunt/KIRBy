# mcs-pairs-carry-no-ordering-hash

**title:** The local MCS pair files record nothing about which molecule list they were numbered against

**area:** tests/experiments/similarity_metrics/ — every .py file in the directory

**class:** A-wrong-answer

**severity as reported:** large

**location:** tests/experiments/similarity_metrics/mcs_cache.py:72-73; consumers at aggregate.py:169, ranker_experiment.py:363, feature_ablation.py:367 and :410, similarity_noise_cleaning.py:289, creative_hybrids.py:128, creative_hybrids_v2.py:186, winning_recipe_reselected.py:66, winning_recipe_vs_standard.py:113, triplet_similarity.py:183, nary_similarity.py:155, shape_similarity.py:117, aug3d_similarity.py:137, extra_fingerprints.py:159, structural_embeddings.py:143, augmented_descriptor_reps.py:123

## scenario as reported
Run 1: `python mcs_cache.py herg_ki` samples 8000 molecule pairs off load_prepared('herg_ki') and writes mcs_pairs_herg_ki.npz holding I, J (row numbers), mcs and dy. Run 2 is any later day on which sim_herg_ki.npz has been rebuilt — build_local_datasets.py re-reads ~/.cache/kirby/chembl_herg_ki.csv and overwrites the npz, so a CSV that gained rows, lost rows, or changed row order produces a different molecule ordering under the same file name. Every one of the 16 read sites above then loads mcs_pairs_herg_ki.npz by dataset name alone, takes D[I, J] over the NEW molecules, and correlates it against the OLD pairs' MCS distances. Nothing raises: I and J stay inside the array as long as the molecule count did not shrink. ranker_experiment.py:365 additionally reads `dy` from that file and uses it as the activity target the feature ranker selects on, so the selection is made against the old labels and scored against the new ones. Every structural fidelity number in this directory is that one Spearman.

## evidence as reported
mcs_cache.py:72-73 — `out = os.path.join(HERE, f'mcs_pairs_{name}.npz')` then `np.savez_compressed(out, I=I, J=J, mcs=mcs, dy=dy)`. Loading all three files on disk confirms the stored keys are exactly ['I','J','mcs','dy'], with max(I,J) = n-1 in each case (herg_ki 1414 of 1415, logd 1426 of 1427, adme_solubility 2172 of 2173). The framework's own pair cache does carry the guard: src/kirby/similarity/mcs.py:117-129 `_ordering_hash`, stored at :105-106, checked on every read at :210-222, and refused for a hash-less file by `assert_pairs_match_smiles` at :180-185. No file in tests/experiments/similarity_metrics/ imports kirby.similarity.mcs — grep for `from kirby` in the directory returns only representations.molecular and datasets.flab/graphinity. mcs_cache.py's own docstring lines 26-30 states these files predate smiles_sha.

## fix as proposed
Store the ordering hash beside the pairs: import `pairs_ordering_hash` from kirby.similarity.mcs and add `smiles_sha=pairs_ordering_hash(p['smiles'])` to the savez at mcs_cache.py:73, plus `n_pairs` and `seed`. Give the directory one loader (in _data.py) that every script calls instead of its own np.load, and have it raise when the stored hash differs from the hash of load_prepared(name)['smiles']. The three files on disk have no hash, so they must be rebuilt once — 8000 pairs at about 0.09 seconds per pair is roughly 12 minutes per dataset. While rebuilding, note that `--pairs` (mcs_cache.py:81) changes the pair count but not the file name, so a run with a smaller `--pairs` silently overwrites the 8000-pair set.
