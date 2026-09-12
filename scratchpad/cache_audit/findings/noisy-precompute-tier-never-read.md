# noisy-precompute-tier-never-read

**title:** The noisy hybrid precompute writes a search lineage no experiment ever asks for; n15 rebuilds the same searches under a different tier

**area:** tests/experiments/noise/ (n01–n15, noise_helpers.py, _similarity_hybrid_space.py, the _smoke_* files) plus src/kirby/noise_spec.py and src/kirby/utils/noise_estimation.py

**class:** B-never-hits

**severity as reported:** large

**location:** tests/experiments/noise/noise_helpers.py:368-391; tests/experiments/noise/n15_clean_vs_noisy_hybrid.py:122-128; tests/experiments/_shared.py:4991-5002; src/kirby/find_best_hybrid.py:7326-7333

## scenario as reported
run_all('logd', ...) reaches noise_helpers.py:369, builds 3 NoiseSpecs from noisy_search_specs() and calls precompute_hybrid_searches(..., label_noise=spec) for each. Inside find_best_hybrid, tier becomes 'general__noise_gaussian_e0050_s0_spread' (find_best_hybrid.py:7326) and the search pickle is written to searches/logd__general__noise_gaussian_e0050_s0_spread__none.pkl. n15 is the only experiment that ever asks for a noisy blueprint, and _fetch_bp always passes score_on_noisy_labels=True (n15:128). That makes _shared.py:4991 set _noisy_selection=True, so main_dict is left None and the in-memory data['precomputed_searches_noisy'][sid] dict is skipped; the disk fallback then re-enters find_best_hybrid, which at line 7332 appends '__selnoisy' to the tier and asks for searches/logd__general__noise_gaussian_e0050_s0_spread__selnoisy__none.pkl. Different filename, clean miss. The precompute's 3 (or 9, with augmentations) full searches are paid, stored, and read by nothing; n15 then pays a second full search per level.

## evidence as reported
noise_helpers.py:367 'Cost: |NOISE_SPECS| x the clean precompute.'; _shared.py:4991 '_noisy_selection = bool(kwargs.get(\'score_on_noisy_labels\'))' and :4994 'if data is not None and not _geo_request and not _noisy_selection:'; find_best_hybrid.py:7327-7333 'if score_on_noisy_labels: _noise_id = f\'{_noise_id}__selnoisy\' ... tier = f\'{tier}__selnoisy\''; _search_cache_path (find_best_hybrid.py:153-161) builds the stem from dataset_name + tier + aug_strategy only. precompute_hybrid_searches (_shared.py:4471-4493) never passes score_on_noisy_labels.

## fix as proposed
Decide which noisy lineage the suite wants and build exactly that one. Either pass score_on_noisy_labels=True through precompute_hybrid_searches when label_noise is set (so the precompute writes the '__selnoisy' tier n15 reads), or drop the noisy precompute block at noise_helpers.py:368-391 and let n15 build the lineage itself. Whichever is chosen, the in-memory dict key must carry the same marker as the tier, or the dict stays unreachable.
