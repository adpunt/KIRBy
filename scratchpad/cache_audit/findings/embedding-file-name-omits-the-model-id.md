# embedding-file-name-omits-the-model-id

**title:** The transformer model is an environment variable; the cached embedding file name keeps only the alias

**area:** tests/experiments/similarity_metrics/ — every .py file in the directory

**class:** A-wrong-answer

**severity as reported:** small

**location:** tests/experiments/similarity_metrics/learned_embeddings.py:35-36, 128-134; read back at run_battery_moleculeace.py:88-90

## scenario as reported
Run 1: `python learned_embeddings.py` embeds every molecule with seyonec/ChemBERTa-zinc-base-v1 and writes emb_CHEMBL234_Ki_chemberta.npy. Run 2: `CHEMBERTA_ID=DeepChem/ChemBERTa-77M-MLM python learned_embeddings.py` — line 128 builds the same file name, line 129 finds it, prints `cached`, and continues. The second model never runs. run_battery_moleculeace.py:89-90 then loads that file and scores the FIRST model's embedding under both the cosine and euclidean rows of a battery the user believes is measuring the second model.

## evidence as reported
learned_embeddings.py:35-36 — `CHEMBERTA = os.environ.get('CHEMBERTA_ID', 'seyonec/ChemBERTa-zinc-base-v1')` and the same for MOLFORMER_ID. learned_embeddings.py:128 — `out = os.path.join(args.data_dir, f'emb_{name}_{which}.npy')`, where `which` is the loop variable over ('chemberta', 'molformer'), never `mid`. learned_embeddings.py:129-130 — `if os.path.exists(out): print(f'  cached {out}'); continue`. The model id at :131 is only read after the existence check.

## fix as proposed
Put the model id in the file name: `emb_{name}_{which}_{sha1(mid)[:8]}.npy`, or write the id into a sidecar and refuse a file whose stored id differs. Existing emb_*.npy files were written with the two defaults, so they can be renamed rather than recomputed.
