"""Forward-compatibility representation list for c13.

WHAT THIS LIST IS. c13 asks one question: when a powerful new set of features is
published, can the hybrid search take it in and use it? It answers it by taking ONE
new representation at a time, searching a hybrid over the dataset's pool without it,
searching again with it added, and subtracting. A Gaussian matrix of the same number
of features is the selection-bias control.

So this list is NOT the pruned ten. The pruned ten in DATASET_REPS['general'] is the
pool a candidate is added TO — the "without" side of the subtraction. This list is the
candidates themselves, and nothing on it is in any dataset's pool.

ADMISSION RULES. A model is on this list only if all five hold:
  1. It is a model whose learned embeddings are usable as a fixed feature block.
     Not a fingerprint, not a descriptor calculator, not a k-mer count.
  2. Its paper or its weight release is 2023 or later.
  3. Its weights are publicly downloadable. The URL is in the entry.
  4. There is a published number, a named comparator, and the comparator's number.
  5. It emits a fixed-length vector per input entity. `embedding` says how.

Everything cut is recorded in CUT at the bottom of each domain block with the rule it
failed, so an absence is never silent.

PROVENANCE. Compiled 2026-09-10 from a seven-lane literature search, each candidate then
adversarially re-checked by a second reader who opened the paper, the weights page and the
evidence page independently. `checked` records what that second reader confirmed. Where a
number could not be confirmed, the entry says so in the same field as the number. Nothing
here is a measured KIRBy result — every number is quoted from the cited source.

Fields:
  model        the published model name
  year, venue  first release (arXiv v1 or weights, whichever is earlier) and where
  paper        URL
  weights      URL, and the license where it is stated on the page
  embedding    what one row is, and how it is pooled
  code         NEW if a builder has to be written; otherwise the KIRBy function and
               whether it has ever built successfully
  evidence     metric, number, comparator, comparator's number, benchmark
  source       URL the evidence was read from
  distinct     what its pretraining or architecture does that the pool does not
  checked      what the second reader confirmed or could not confirm
  flag         anything that changes what a run of it would mean
"""

# =============================================================================
# MOLECULAR — logd, caco2, herg, herg_ki, qm9, adme_hlm, adme_rlm, adme_mdr1
# =============================================================================

MOLECULAR = {
    'chemeleon': {
        'model': 'CheMeleon',
        'year': 2025, 'venue': 'arXiv 2506.15792 (v1 Jun 2025, v2 Feb 2026); ships in Chemprop 2.2.0',
        'paper': 'https://arxiv.org/abs/2506.15792',
        'weights': 'https://doi.org/10.5281/zenodo.15426600 — MIT code, CC0 weights; '
                   'CheMeleonFingerprint extractor at https://github.com/JacksonBurns/CheMeleon',
        'embedding': 'One row per molecule, 2048 features. D-MPNN, message-passing width 2048, '
                     '6 iterations, per-atom hidden states mean-aggregated. The repo ships a '
                     'frozen extractor that takes SMILES and returns the vector without fine-tuning.',
        'code': 'NEW',
        'evidence': 'Win rate over 28 Polaris tasks: CheMeleon 75%, Random Forest 68%, fastprop 36%, '
                    'Chemprop 32%. Over 30 MoleculeACE assays: CheMeleon 97%, Random Forest 50%; '
                    '100% on the activity-cliff subset.',
        'source': 'https://arxiv.org/abs/2506.15792',
        'distinct': 'Pretrained by regressing 1M PubChem molecules onto their computed Mordred '
                    'descriptors, not by masked tokens or contrastive views. mordred is in every '
                    'molecular pool as a hand-computed block; this is that signal learned into a '
                    'message-passing encoder. No D-MPNN is in the repo at all.',
        'checked': 'Both win-rate pairs confirmed in the v2 abstract. v1 quoted 79/46/39/36 for the '
                   'same four models; the v2 numbers above are the current ones.',
        'flag': 'The only 2025+ molecular candidate whose evidence is on ADMET-style regression, '
                'which is what logd, caco2 and the three adme_* datasets are. Because it is trained '
                'on Mordred targets and mordred sits in the pool it is added to, its c13 delta '
                'measures redundancy against a block that is already there.',
    },
    'mol_jepa': {
        'model': 'Mol-JEPA (Boehringer Ingelheim)',
        'year': 2026, 'venue': 'arXiv 2608.22642, Aug 2026',
        'paper': 'https://arxiv.org/html/2608.22642v2',
        'weights': 'https://huggingface.co/Flogrammer/Mol-JEPA — CC BY-NC 4.0 (weights), CC BY 4.0 (paper)',
        'embedding': 'One row per molecule, 512 features from the learnable CLS token. Per-modality '
                     'latents are also exposed at [batch, 13, 512] and can be mean-pooled to 512 instead.',
        'code': 'NEW',
        'evidence': 'ExpansionRx Caco-2 permeability MAE: Mol-JEPA 0.34 +/- 0.02, CLAMP 0.40 +/- 0.05, '
                    'Chemprop 0.39 +/- 0.04. It loses ASAP MERS-CoV-2: 0.59 +/- 0.10 against '
                    'Chemprop 0.52 +/- 0.08.',
        'source': 'https://arxiv.org/html/2608.22642v2',
        'distinct': 'Masks whole modalities and predicts their latent representations, across 14 '
                    'molecular modalities fused by one transformer. Nothing in the repo is a '
                    'joint-embedding-predictive model; unimol and schnet are 3D-only, chemberta and '
                    'molformer are SMILES-only.',
        'checked': 'Both number pairs confirmed on the arXiv HTML.',
        'flag': 'The only candidate with a published number on Caco-2 permeability, which is one of '
                'the eight molecular datasets. Non-commercial weights license.',
    },
    'glacier': {
        'model': 'GLACIER',
        'year': 2026, 'venue': 'arXiv 2606.11382, Jun 2026',
        'paper': 'https://arxiv.org/html/2606.11382v1',
        'weights': 'https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol — MIT',
        'embedding': 'One row per molecule, 512 features in a shared multimodal space, from an '
                     'attentive-aggregation readout (a learned weighted average over atoms), not a mean.',
        'code': 'NEW',
        'evidence': 'Mean AUROC over 9 classification tasks: GLACIER (MiniMol teacher) 0.799, '
                    'MiniMol 0.781, MolFormer 0.757, ChemBERTa-100M 0.726. Mean RMSE over 2 '
                    'regression tasks: GLACIER 0.806, MiniMol 1.040, MolFormer 1.383.',
        'source': 'https://arxiv.org/html/2606.11382v1',
        'distinct': 'Student-teacher distillation across a graph encoder, a SMILES text encoder and a '
                    'tabular encoder over 217 RDKit descriptors, aligned contrastively. Nothing in the '
                    'repo fuses a descriptor tower with a graph tower inside one pretrained model.',
        'checked': 'Confirmed on the arXiv HTML.',
        'flag': 'Its teacher is MiniMol, which is implemented below. Running both separates the '
                'distillation from the model it distils.',
    },
    'clamp': {
        'model': 'CLAMP (Contrastive Language-Assay Molecule Pre-training)',
        'year': 2023, 'venue': 'ICML 2023',
        'paper': 'https://arxiv.org/abs/2303.03363',
        'weights': 'https://github.com/ml-jku/clamp — GPL-3.0 (the second reader corrected an MIT '
                   'claim); mirror at https://huggingface.co/scikit-fingerprints/clamp',
        'embedding': 'One row per molecule, 768 features. create_clamp docstring states (n, 768).',
        'code': 'create_clamp, molecular.py:3568. BUILDS — measured 2026-09-10 on caco2 locally: '
                '768 features over 2158 molecules in 21.3s, checkpoint loaded from '
                'src/clamp/data/models/clamp_clip/checkpoint.pt. It is in reps_failed as drug_clamp '
                'on kiba and bindingdb_kd (2026-04-21), so the DTA failure is environment, not a '
                'missing builder.',
        'evidence': 'Praski et al., 25 models x 25 datasets, hierarchical Bayesian test: CLAMP mean '
                    'AUROC 82.55% at mean rank 5.40, ECFP 79.89% at mean rank 7.52. The only one of '
                    'the 25 the test calls significantly better than ECFP.',
        'source': 'https://arxiv.org/html/2508.06199v2',
        'distinct': 'Pretrained on molecule-bioassay text pairs by contrastive learning, so its '
                    'features carry assay-level bioactivity context. Nothing else is trained on assay text.',
        'checked': 'Table 1 numbers confirmed. Table 2 text confirmed: "Only the CLAMP model is '
                   'statistically significantly better than ECFP." A posterior probability of 0.86 '
                   'was reported in the first pass and is NOT printed in the paper — the paper states '
                   'the decision, not that number.',
        'flag': 'The same benchmark describes CLAMP\'s molecule side as "the concatenation of three '
                'molecular fingerprints (ECFP and RDKit) processed through a two-layer MLP". Both '
                'ecfp4 and rdkfp are already in the repo, so its 768 features are a learned '
                'projection of blocks the pool can already hold.',
    },
    'ouroboros': {
        'model': 'Ouroboros',
        'year': 2026, 'venue': 'Advanced Science 13(17), 4 Jan 2026',
        'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13042461/',
        'weights': 'see the paper\'s availability statement',
        'embedding': 'One row per molecule, 2048 features. The paper states "a representation module '
                     'of 1D molecular representation with a size of 2048 is pre-trained."',
        'code': 'NEW',
        'evidence': 'Mean over ten molecular property regression tasks: Ouroboros 0.687, Uni-Mol 0.676. '
                    'Mean Spearman across NCI/DTP cell lines: Ouroboros 0.657, GeminiMol 0.607.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13042461/',
        'distinct': 'Learns a conformational space and pharmacophore into the molecular foundation model.',
        'checked': 'All three quoted sentences confirmed verbatim.',
        'flag': 'Self-report, and 0.687 is a FINE-TUNED number. c13 runs frozen features, so this '
                'number does not describe the block c13 would build.',
    },
    'chemfm': {
        'model': 'ChemFM (1B / 3B, LLaMA-based)',
        'year': 2025, 'venue': 'PMC12764477',
        'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12764477/',
        'weights': 'see the paper\'s availability statement',
        'embedding': 'One row per molecule; pooled decoder hidden states.',
        'code': 'NEW',
        'evidence': 'Two sources that disagree, and the disagreement is the point. Own paper: '
                    '"ChemFM-3B reduced root mean squared errors by 0.039 on ESOL, 0.245 on FreeSolv, '
                    '0.010 on Lipophilicity, and 0.024 on PDBbind", and on ADMET "an average '
                    'improvement of approximately 7.09%" over 13 classification and 9 regression sets. '
                    'Independent, Praski Table 1: ChemFM mean rank 12.60, mean AUROC 78.25%, BELOW '
                    'ECFP at 7.52 / 79.89%.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12764477/ and https://arxiv.org/html/2508.06199v2',
        'distinct': 'A 1B/3B decoder LM on SMILES. The repo has no model of that scale.',
        'checked': 'Both sides confirmed. The good numbers are fine-tuned; the frozen-feature number '
                   'is below the fingerprint baseline. c13 measures the frozen one.',
        'flag': 'Carries a published NEGATIVE frozen-feature result. That is a calibration point for '
                'c13, not a reason to leave it off.',
    },
    'mist': {
        'model': 'MIST (28M / 1.8B)',
        'year': 2025, 'venue': 'arXiv 2510.18900',
        'paper': 'https://arxiv.org/html/2510.18900v2',
        'weights': 'see the paper\'s availability statement',
        'embedding': 'One row per molecule; pooled transformer hidden states.',
        'code': 'NEW',
        'evidence': 'Fine-tuned AUROC 0.915 against the Lee et al. GNN at 0.894. Discordant triplets '
                    'resolved: MIST 64.3%, the same GNN 50%. DREAM Olfaction 2024 Pearson: MIST '
                    '0.62 +/- 0.05 against a top-submission ensemble at 0.57. Validation R2 0.76 '
                    '(28M) and 0.80 (1.8B).',
        'source': 'https://arxiv.org/html/2510.18900v2',
        'distinct': 'Pretrained on olfactory and organoleptic property data, a supervision signal no '
                    'other molecular model in the repo has seen.',
        'checked': 'All four numbers confirmed verbatim.',
        'flag': 'Every published number is on odour or organoleptic tasks. There is no ADMET or '
                'physicochemical number for it, so the eight molecular datasets are out-of-domain '
                'for its evidence.',
    },
    'coati': {
        'model': 'COATI (multimodal E(3)-equivariant)',
        'year': 2024, 'venue': 'JCIM',
        'paper': 'https://arxiv.org/html/2508.06199v2 (its own paper is paywalled)',
        'weights': 'see the paper\'s availability statement',
        'embedding': 'One row per molecule.',
        'code': 'NEW',
        'evidence': 'Praski Table 1: COATI mean rank 13.92, mean AUROC 78.02%, against ECFP 7.52 / 79.89%.',
        'source': 'https://arxiv.org/html/2508.06199v2',
        'distinct': 'E(3)-equivariant multimodal encoder; the repo has no equivariant molecule model '
                    'except schnet.',
        'checked': 'Confirmed in Table 1. Its own paper could not be reached (paywalled), so there is '
                   'NO positive paired number for it anywhere in this dossier — the only number is negative.',
    },
    'unimol_v2': {
        'model': 'Uni-Mol2',
        'year': 2024, 'venue': 'NeurIPS 2024',
        'paper': 'https://arxiv.org/abs/2508.06199 (benchmark entry)',
        'weights': 'pip install unimol-tools; weights at https://huggingface.co/dptech/Uni-Mol2',
        'embedding': 'One row per molecule, 768 features. Two-track transformer over coordinates, '
                     'pair distances and the graph.',
        'code': 'create_unimol_v2, molecular.py. BUILT SUCCESSFULLY as drug_unimol_v2 on kiba and '
                'bindingdb_kd (2026-04-21). ⛔ HANGS on caco2 locally — measured 2026-09-10: it '
                'loops "Start generating conformers..." over the 2158 molecules repeatedly and then '
                'sits at 0% CPU indefinitely, under OMP_NUM_THREADS=1 KIRBY_LGBM_THREADS=1. It had '
                'to be killed after roughly 50 minutes. unimol_tools spawns its own worker pool for '
                'conformer generation, so a single-thread OpenMP setting is the suspect but that is '
                'NOT confirmed. Never completed on a molecular dataset.',
        'evidence': 'Praski Table 1: Uni-Mol2 mean rank 13.32, mean AUROC 78.11%, against ECFP '
                    '7.52 / 79.89%.',
        'source': 'https://arxiv.org/html/2508.06199v2',
        'distinct': '3D two-track transformer. unimol (v1) is in the repo but in no molecular pool either.',
        'checked': 'Confirmed. The only independent benchmark number for it is BELOW a fingerprint '
                   'that is already in every molecular pool.',
        'flag': 'The only molecular candidate on this list proven to build on the cluster.',
    },
    'minimol': {
        'model': 'MiniMol (GINE GNN, Graphcore)',
        'year': 2024, 'venue': 'Graphcore Research',
        'paper': 'https://github.com/graphcore-research/minimol',
        'weights': 'pip install minimol — github.com/graphcore-research/minimol. NOTE the second '
                   'reader found huggingface.co/graphcore/minimol returns 401 and does not exist.',
        'embedding': 'One row per molecule, width set by the checkpoint.',
        'code': 'create_minimol, molecular.py. WRITTEN, NEVER BUILT — in reps_failed on kiba and '
                'bindingdb_kd (2026-04-21).',
        'evidence': 'From the GLACIER table above: MiniMol mean AUROC 0.781 over 9 classification '
                    'tasks against MolFormer 0.757 and ChemBERTa-100M 0.726; mean RMSE 1.040 over 2 '
                    'regression tasks against MolFormer 1.383.',
        'source': 'https://arxiv.org/html/2606.11382v1',
        'distinct': 'A small pretrained GINE graph network. The repo\'s other GNNs (grover, graphmvp, '
                    'molclr, mhg_gnn) are all differently pretrained.',
        'checked': 'The year was not confirmed against a paper. The weights URL in the first pass was '
                   'wrong and is corrected above.',
    },
}

MOLECULAR_CUT = {
    'rmat': 'Published before 2023. arXiv v1 is 12 Oct 2021; the "2024" in the first pass '
            'corresponds to nothing findable. (create_rmat exists at molecular.py:3658 and is in '
            'reps_failed on kiba/bindingdb_kd.) Its Praski rank is 6.08 at 80.83% mean AUROC against '
            'ECFP 7.52 / 79.89% — second of 25, but the paper\'s own Bayesian test does not call it '
            'a difference. https://arxiv.org/html/2508.06199v2',
    'mole': 'Weights not downloadable. github.com/recursionpharma/mole_public says "Download '
            'pretrained models: TODO", has no releases and no checkpoint file in its tree. Also '
            'arXiv v1 is 3 Nov 2022, before the bar. (create_mole exists at molecular.py:4170.) Its '
            'headline — beats the best published result on 10 of 22 TDC ADMET tasks, Nature '
            'Communications s41467-024-53751-y — is a fine-tuning result, not a frozen-feature one.',
}

# =============================================================================
# ANTIBODY — the 19 antibody datasets
# =============================================================================

ANTIBODY = {
    'ism': {
        'model': 'ISM (Implicit Structure Model), ism_t33_650M_uc30pdb',
        'year': 2025, 'venue': 'ICLR 2025',
        'paper': 'https://www.biorxiv.org/content/10.1101/2024.11.08.622579v2.full',
        'weights': 'https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb — MIT',
        'embedding': 'One row per antibody. ESM-2 650M architecture, 1280 features per residue; '
                     'mean-pool residues per chain, concatenate heavy and light = 2560 features. '
                     'Drop-in for the existing esm2 loader.',
        'code': 'NEW (but the esm2 loader path is reusable — same architecture and width)',
        'evidence': 'FLAb2 (Chungyoun & Gray, 30 models, 240 datasets, 4M antibodies, 32 studies): '
                    'ISM 650M has the highest count of statistically significant zero-shot Spearman '
                    'correlations of any of the 30 — 94 significant correlations, 39% of the '
                    'developability datasets. Few-shot: ISM 3B ties ESM2 3B as best overall at '
                    'rho 0.56; thermostability rho 0.61.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12767642/',
        'distinct': 'Same weights shape as esm2, different pretraining objective: ESM-2 '
                    'continued-trained to predict discrete structure tokens from a per-residue '
                    'microenvironment autoencoder. Unlike saprot it needs no 3Di token at inference.',
        'checked': 'The 94 / 39% sentence confirmed verbatim. The second reader marked the evidence '
                   'field not-fully-confirmed because the few-shot rho values were not located in the '
                   'text opened.',
        'flag': 'Best-evidenced antibody candidate on this list, and the benchmark behind it is '
                'third-party over 240 datasets rather than a self-report.',
    },
    'igbert': {
        'model': 'IgBert (Exscientia, 420M)',
        'year': 2024, 'venue': 'PLOS Computational Biology 2024 / arXiv 2403.17889',
        'paper': 'https://arxiv.org/html/2403.17889v1',
        'weights': 'https://huggingface.co/Exscientia/IgBert — CC BY 4.0',
        'embedding': 'One row per antibody, 1024 features. Paired model takes heavy and light as one '
                     'sequence; the paper averages over both chains\' residue tokens.',
        'code': 'embed_igbert and embed_igbert_cls exist in antibody.py:443 and :479 but NEITHER is '
                'a key in ANTIBODY_REPS, so no pool can request them. WARNING: '
                'complexity_helpers.py:338 and :395 record igbert and igbert_cls as dead/broken reps '
                'excluded from the abagym and humanness surveys. Registering the name is NOT the '
                'whole cost — find out why they were called broken first.',
        'evidence': 'Binding affinity R2, Koenig anti-VEGF (N=4275): IgBert 0.174 +/- 0.032, '
                    'IgT5 0.25 +/- 0.019, AbLang 0.244 +/- 0.034, ProtT5 0.206 +/- 0.029. '
                    'Shanehsazzadeh (N=422): IgBert 0.306 +/- 0.114, AbLang 0.293 +/- 0.117, '
                    'IgT5 0.274 +/- 0.070. Paired sequence recovery VH 91.29% / VL 96.47% against '
                    'AbLang 91.05 / 94.93 and AntiBERTy 89.98 / 95.53.',
        'source': 'https://arxiv.org/html/2403.17889v1',
        'distinct': 'igt5 is in the pools, igbert is not. Same OAS paired training data, different '
                    'backbone (420M BERT against 3B T5), and the loser on Koenig binding — which is '
                    'exactly the head-to-head c13 measures, on a dataset the repo has '
                    '(flab_binding_koenig).',
        'checked': 'Every number confirmed against the arXiv HTML.',
    },
    'mint': {
        'model': 'MINT (Multimeric INteraction Transformer, 650M)',
        'year': 2026, 'venue': 'Nature Communications 2026 (bioRxiv 2025.03.09.642188)',
        'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11956943/',
        'weights': 'https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt — CC BY 4.0',
        'embedding': 'One row per antibody. ESM-2 backbone, 1280 features per residue averaged over '
                     'length per chain; with sep_chains=True the wrapper returns heavy and light '
                     'concatenated = 2560 features.',
        'code': 'embed_mint exists in antibody.py:921 with a checkpoint loader in protein.py, and '
                "'mint': 1280 sits in a dimension table at antibody.py:1613 — but 'mint' is not a "
                'key in ANTIBODY_REPS. Same WARNING as igbert: complexity_helpers.py:338 and :395 '
                'record mint as dead/broken.',
        'evidence': 'On FLAb (four datasets) MINT beats IgBert and IgT5 on all four, by more than 10% '
                    'on three. SKEMPI ddG on mutation: 29% improvement over ESM-2 650M/3B, ProGen 3B '
                    'and ProtT5 3B. SARS-CoV-2 m396 binding: 14% higher Spearman than AbMap at 0.5% '
                    'of the samples.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11956943/',
        'distinct': 'Pretrained on 96M protein-protein interaction pairs from STRING with cross-chain '
                    'attention. ablang2 and igt5 saw only antibody heavy/light pairs; MINT saw general '
                    'interacting pairs and encodes the two chains jointly.',
        'checked': 'All three sentences confirmed verbatim in the bioRxiv full text. The paper gives '
                   'these as RELATIVE gains — no per-dataset absolute values are in the text, they '
                   'are in figures.',
        'flag': 'ddg_combined is a ddG-on-mutation dataset and SKEMPI is the benchmark MINT reports on.',
    },
    'abcdr_esmc': {
        'model': 'AbCDR-ESMC (CDR-masked ESM-C 600M)',
        'year': 2026, 'venue': 'Communications AI & Computing 2026 (bioRxiv 2025.10.31.685149)',
        'paper': 'https://www.biorxiv.org/content/10.1101/2025.10.31.685149v1.full',
        'weights': 'https://huggingface.co/NOC-Lab/AbCDR-ESMC — MIT',
        'embedding': 'One row per antibody, 1152 features. ESM-C 600M backbone, mean-pooled over all '
                     'positions of both chains. The model card gives the exact call.',
        'code': 'NEW (esmc loader path is reusable)',
        'evidence': 'Binding affinity R2, Table 2: AbCDR-ESMC G6 single mutants 0.313, D44 0.345, '
                    'Trastuzumab 0.350; AbCDR-ESM2 G6 0.298, D44 0.359, Trastuzumab 0.350; '
                    'IgT5 G6 0.250, D44 0.297. Table 3, anti-Fluorescein combinatorial mutants: '
                    'AbCDR-ESM2 0.693 against base ESM2 0.547.',
        'source': 'https://www.biorxiv.org/content/10.1101/2025.10.31.685149v1.full',
        'distinct': 'esmc is in the pools; this is that backbone continued-trained on paired variable '
                    'domains with masking concentrated in the six CDR loops instead of uniformly.',
        'checked': 'AbCDR-ESMC 0.313 and the 0.693/0.547 pair confirmed exactly from Tables 2 and 3. '
                   'An IgT5 value of 0.264 quoted in the first pass is NOT in the table — the table '
                   'gives IgT5 0.250 on G6 and 0.297 on D44.',
        'flag': 'The preprint said checkpoints were private until acceptance; the second reader found '
                'the NOC-Lab repo public.',
    },
    'antifold': {
        'model': 'AntiFold (inverse folding, OPIG)',
        'year': 2025, 'venue': 'Bioinformatics Advances 5(1) 2025',
        'paper': 'https://academic.oup.com/bioinformaticsadvances/article/5/1/vbae202/8090019',
        'weights': 'https://github.com/oxpig/AntiFold — BSD 3-Clause',
        'embedding': 'One row per antibody, but it needs a 3D structure, so an ABodyBuilder model has '
                     'to exist first. The repo already builds those for the structural rep. AntiFold '
                     'emits per-residue amino-acid log-probabilities plus hidden states; pool over '
                     'CDR residues or all residues.',
        'code': 'NEW',
        'evidence': 'Warszawski anti-lysozyme binding affinity Spearman: AntiFold 0.418, ESM-IF1 '
                    '0.334, AbMPNN 0.322, ProteinMPNN 0.301. CDRH3 sequence recovery: AntiFold 60%, '
                    'AbMPNN 56%, ESM-IF1 43%, ProteinMPNN 35%.',
        'source': 'https://academic.oup.com/bioinformaticsadvances/article/5/1/vbae202/8090019',
        'distinct': 'An inverse-folding model: it scores sequence given backbone geometry, the '
                    'opposite conditioning to every language model in the pools. saprot and prosst '
                    'take structure as an extra token track but still predict sequence from sequence.',
        'checked': 'Confirmed to three decimals. A tie shown in the first pass between ESM-IF1 and '
                   'AbMPNN is a rounding artefact — ESM-IF1 is ahead.',
        'flag': 'Needs a built structure per antibody. That cost is already paid wherever the '
                'structural rep runs.',
    },
    'abmpnn': {
        'model': 'AbMPNN',
        'year': 2023, 'venue': 'ICML 2023 Workshop on Computational Biology / arXiv 2310.19513',
        'paper': 'https://arxiv.org/abs/2310.19513',
        'weights': 'https://zenodo.org/records/8164693 — license not stated on the Zenodo record',
        'embedding': 'One row per antibody, structure required. Message-passing GNN over backbone '
                     'atoms; per-residue decoder features pooled over CDR residues or all residues, '
                     'plus 20 amino-acid log-probabilities per position.',
        'code': 'NEW',
        'evidence': 'From the AntiFold comparison (third-party): AbMPNN CDRH3 sequence recovery 56% '
                    'against ProteinMPNN 35%, ESM-IF1 43%, AntiFold 60%. Warszawski anti-lysozyme '
                    'binding Spearman: AbMPNN 0.322, ProteinMPNN 0.301, AntiFold 0.418. Remaining '
                    'CDRs 63-76% against AntiFold 75-84%.',
        'source': 'https://academic.oup.com/bioinformaticsadvances/article/5/1/vbae202/8090019',
        'distinct': 'A graph network on backbone coordinates, not a transformer over sequence tokens.',
        'checked': 'All figures confirmed from an independent comparator paper rather than its own report.',
        'flag': 'It loses to AntiFold on both published metrics. Running both is what separates '
                '"inverse folding helps" from "this inverse folder helps".',
    },
    'p_iggen': {
        'model': 'p-IgGen (17.3M autoregressive decoder, OPIG)',
        'year': 2024, 'venue': 'Bioinformatics 40(11) 2024',
        'paper': 'https://academic.oup.com/bioinformatics/article/40/11/btae659/7888884',
        'weights': 'https://github.com/oxpig/p-IgGen — BSD 3-Clause',
        'embedding': 'One row per antibody, 768 features. GPT-2-style decoder, 3 layers, 17,349,888 '
                     'params. Heavy and light fed as one string; mean-pool the final hidden states.',
        'code': 'NEW',
        'evidence': 'Zero-shot immunogenicity (anti-drug antibody response, 217 therapeutics) Pearson: '
                    'p-IgGen 0.53, developable p-IgGen 0.52, ProGen/small 151M 0.48, ProGen/medium '
                    '764M 0.46, IgGen 0.45. VH-VL mutation-rate correlation 0.52 against 0.51 for '
                    'natural sequences and -0.04 for randomly paired.',
        'source': 'https://academic.oup.com/bioinformatics/article/40/11/btae659/7888884',
        'distinct': 'The only autoregressive decoder in the candidate set and by far the smallest at '
                    '17M params, so its features come from left-to-right likelihood rather than '
                    'masked reconstruction.',
        'checked': 'Table 2 confirmed exactly, including the parameter count and the three attention layers.',
        'flag': 'Its evidence task is immunogenicity over therapeutics, which is the closest published '
                'match to humanness_therapeutic of anything on this list.',
    },
    'antiberta2': {
        'model': 'AntiBERTa2 / AntiBERTa2-CSSP (Alchemab, 202M RoFormer)',
        'year': 2023, 'venue': 'MLSB 2023',
        'paper': 'https://huggingface.co/alchemab/antiberta2',
        'weights': 'https://huggingface.co/alchemab/antiberta2 — NON-COMMERCIAL ONLY per the model card',
        'embedding': 'One row per antibody. 16 layers, 1024 features per residue; mean-pool per chain '
                     'and concatenate = 2048.',
        'code': 'NEW',
        'evidence': 'AbLingua head-to-head, CDRH3 binder recall on the validation set: AntiBERTa2 + FC '
                    '54.5%, against AbLingua-1.7B + ResNet 58.0%, ESM2 + FC 57.0%, IgBert + ResNet '
                    '56.3%, IgT5 + ResNet 55.2%, AbLang2 + ResNet 54.0%, AntiBERTy + FC 52.5%.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13484476/',
        'distinct': 'Rotary-embedding RoFormer on OAS plus SAbDab; the CSSP variant adds a contrastive '
                    'structure objective. abroberta is the only other antibody-only encoder in the '
                    'pools and it is 768-wide RoBERTa on a different corpus.',
        'checked': 'Confirmed exactly, and it is a third-party benchmark rather than Alchemab\'s own — '
                   'a measured mid-table finish, not a claimed win.',
        'flag': 'Non-commercial license. Check that against how the results get published.',
    },
    'currab': {
        'model': 'CurrAb (curriculum-trained paired antibody LM, 650M)',
        'year': 2025, 'venue': 'PLOS Computational Biology 2025',
        'paper': 'https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013473',
        'weights': 'https://huggingface.co/collections/brineylab/curriculum-paper-685b08a4b6986df7c5a5e3c4 '
                   '— license not stated in the article text',
        'embedding': 'One row per antibody, 1280 features. ESM-2 architecture with rotary position '
                     'embeddings; heavy and light trained as one paired sequence, mean-pool over both.',
        'code': 'NEW',
        'evidence': 'Three-way specificity classification (healthy donor vs flu vs CoV) on paired '
                    'sequences: CurrAb more than 10% more accurate than the next best, AbLang2. '
                    'Ahead of IgBERT, ESM-2 and AntiBERTa2 on native-vs-random pair classification.',
        'source': 'https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013473',
        'distinct': 'Trained unpaired-first then annealed to paired sequences on a schedule, rather '
                    'than paired-only (ablang2, igt5) or unpaired-only (antiberty).',
        'checked': 'Both quoted sentences appear in the article. WEAKEST EVIDENCE HERE: every '
                   'per-model accuracy is in figures, not text, so the ">10%" has no pair of raw '
                   'numbers behind it.',
    },
}

ANTIBODY_CUT = {
    'ablingua': 'Weights at the stated URL are not the model the evidence is about. '
                'IDEA-AI4S/AbLingua holds one checkpoint whose config.json reads hidden_size 1280, '
                'num_hidden_layers 30, vocab_size 18950; both weight files are 2.47 GB = 617M params '
                'at fp32, and HuggingFace counts it as 0.6B. The paper\'s numbers are for the 1.7B '
                'model. The NUMBERS ARE CORRECT — CDRH3 binder recall AbLingua-1.7B + ResNet 58.0% '
                'against ESM2 + FC 57.0%, IgBert 56.3%, IgT5 55.2%, AntiBERTa2 54.5%, AbLang2 54.0%, '
                'AntiBERTy 52.5%, paratope F1 75% — so recheck if the 1.7B checkpoint is released. '
                'https://pmc.ncbi.nlm.nih.gov/articles/PMC13484476/',
    'balm': 'Evidence not in the source. Paratope APR as published: BALM 76.2 +/- 0.1, ESM-2 '
            '74.2 +/- 0.4, AntiBERTy 72.8 +/- 1.0, AbLang-H 72.4 +/- 1.0. The first pass claimed '
            '82.1 / 78.2 / 80.1 / 79.4 — not one of those four values is in the table. Redundancy '
            'APR as published: BALM 76.1 +/- 0.5, AbLang-H 66.0, AntiBERTy 64.1, ESM-2 59.5. '
            'Binding affinity Pearson is not given numerically in the text at all. '
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC11128484/',
}

# =============================================================================
# PROTEIN (DTA target side) — davis, kiba, bindingdb_kd
# =============================================================================

PROTEIN = {
    'protrek': {
        'model': 'ProTrek 650M (tri-modal sequence/structure/text, Westlake)',
        'year': 2024, 'venue': 'Nature Biotechnology 2025 (bioRxiv 2024.05.30.596740)',
        'paper': 'https://arxiv.org/html/2506.14796v1 (PFMBench, where it is ranked)',
        'weights': 'https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50',
        'embedding': 'One row per protein, 1024 features. `model.get_protein_repr([seq])` returns '
                     'the contrastive-projection vector; the repo README shows torch.Size([1, 1024]).',
        'code': 'NEW',
        'evidence': 'PFMBench (17 protein foundation models, 38 tasks, adapter tuning): ProTrek ranks '
                    '1st of the 12 core models with a 75% winning rate against ESM-2 650M. ESM-2 '
                    'itself ranks 7th, ESM-C 600M 9th, ESM3 1.4B has a 13% winning rate. Own paper: '
                    'sequence-to-text global retrieval MAP 0.233 against ProtST 0.007 and ProteinDT '
                    '0.004; beats ESM-2 650M on 9 of 11 supervised downstream tasks.',
        'source': 'https://arxiv.org/html/2506.14796v1',
        'distinct': 'Contrastive alignment of sequence against structure against free-text function. '
                    'esm2/esmc/amplify are masked-LMs on sequence only; saprot and prosst use '
                    'structure tokens as extra input, not as a contrastive target.',
        'checked': 'PFMBench ranking and the retrieval MAP confirmed. The second reader marked the '
                   'year field as unconfirmed at 2025 — the preprint is 2024, which still clears the bar.',
    },
    'ism': {
        'model': 'ISM (Implicit Structure Model)',
        'year': 2024, 'venue': 'ICLR 2025 (bioRxiv 2024.11.08.622579)',
        'paper': 'https://www.biorxiv.org/content/10.1101/2024.11.08.622579v2.full',
        'weights': 'https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb — MIT',
        'embedding': 'One row per protein: mean over residues of the 1280-wide final hidden layer. '
                     'Drop-in for ESM-2 650M.',
        'code': 'NEW (same weights as the antibody lane entry — one builder serves both domains)',
        'evidence': 'Long-range contact precision@L: ISM 0.49, ESM-2 650M 0.35, SaProt 0.48. '
                    'Stability change on S669: Spearman 0.53 / AUC 0.76 for ISM against 0.49 / 0.72 '
                    'for ESM-2. Binding residue F1: ISM 0.37, SaProt 0.36.',
        'source': 'https://www.biorxiv.org/content/10.1101/2024.11.08.622579v2.full',
        'distinct': 'Same architecture and sequence-only input as the existing esm2 rep, but the '
                    'weights are fine-tuned to predict structure tokens from an atomic autoencoder. '
                    'Unlike saprot it needs no structure at inference, so it runs on a bare target '
                    'sequence in Davis and KIBA.',
        'checked': 'All three number pairs confirmed. Year field marked unconfirmed at 2025 — bioRxiv '
                   'v1 is Nov 2024.',
        'flag': 'The same checkpoint appears in the antibody list. That makes it the one candidate '
                'where c13 measures the same weights against two different pools.',
    },
    'profluent_e1': {
        'model': 'Profluent E1 (600m / 300m / 150m)',
        'year': 2025, 'venue': 'bioRxiv, 13 Nov 2025',
        'paper': 'https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1.full.pdf',
        'weights': 'https://huggingface.co/Profluent-Bio/E1-600m',
        'embedding': 'One row per protein: mean over residues of the final hidden layer. The paper '
                     'calls it a drop-in replacement for the ESM family; the width is not printed on '
                     'the pages read.',
        'code': 'NEW',
        'evidence': 'ProteinGym substitution assays, average Spearman: E1-600M 0.420, E1-300M 0.416, '
                    'ESM-2 650M 0.414, ESM-2 3B 0.406, ESM-C 600M 0.405, ESM-C 300M 0.406. '
                    'Unsupervised long-range contact precision@L on CAMEO: E1-600M 0.512, ESM-2 3B '
                    '0.434, ESM-C 300M 0.425; on CASP15 0.425 against 0.339 and 0.342. With retrieved '
                    'homologs in context E1-600M reaches 0.477 ProteinGym Spearman against PoET 0.470.',
        'source': 'https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1.full.pdf',
        'distinct': 'Trained on 7 trillion tokens with an optional retrieval-augmented mode that '
                    'conditions on homologs at inference. Nothing in the pool conditions on retrieved '
                    'homologs.',
        'checked': 'Year, weights and every number confirmed.',
    },
    'ankh3': {
        'model': 'Ankh3-Large / Ankh3-XL (T5 encoder-decoder)',
        'year': 2025, 'venue': 'arXiv 2505.20052, May 2025',
        'paper': 'https://arxiv.org/html/2505.20052',
        'weights': 'https://huggingface.co/ElnaggarLab/ankh3-large — CC-BY-NC-SA-4.0',
        'embedding': 'One row per protein: T5 encoder run with the [NLU] prefix, then mean over '
                     'residues. Width not stated on the model card (Ankh-large was 1536).',
        'code': 'NEW',
        'evidence': 'GB1 fitness Spearman: Ankh3-Large 90.30, Ankh-Large 84.0, ESM-2 15B 57.0. '
                    'Fluorescence Spearman: 64.56, 62.0, 56.0. Secondary structure CASP-14 Q3: '
                    'Ankh3-Large 79.28%, Ankh3-XL 82.19%, ESM-2 15B 76.56%. Contact precision@L on '
                    'ProteinNet: Ankh3-XL 60.95%, ESM-2 15B 48.93%.',
        'source': 'https://arxiv.org/html/2505.20052',
        'distinct': 'An encoder-decoder T5 trained with masked denoising at several masking rates plus '
                    'sequence completion, not a BERT-style masked model. No T5-family protein encoder '
                    'is in the DTA pools at all.',
        'checked': 'Year, weights and all four number groups confirmed.',
        'flag': 'Non-commercial share-alike license.',
    },
    'prostt5': {
        'model': 'ProstT5 (bilingual sequence/3Di T5, Rostlab)',
        'year': 2023, 'venue': 'NAR Genomics and Bioinformatics 2024 (preprint 2023)',
        'paper': 'https://arxiv.org/html/2506.14796v1 (PFMBench, where it is ranked)',
        'weights': 'https://huggingface.co/Rostlab/ProstT5',
        'embedding': 'One row per protein: mean over residues of the T5 encoder output, 1024 wide '
                     '(T5-XL encoder width; not confirmed against a config file).',
        'code': 'NEW',
        'evidence': 'PFMBench (17 models, 38 tasks): ProstT5 3B places 2nd of the 12 core models, '
                    'above ProtT5 (3rd), SaProt (6th), ESM-2 650M (7th), ESM-C 600M (9th) and '
                    'ESM3 (12th).',
        'source': 'https://arxiv.org/html/2506.14796v1',
        'distinct': 'A T5 trained to translate between amino acids and Foldseek 3Di in both '
                    'directions, so it emits a structure-informed embedding from sequence alone. The '
                    'DTA pools have no T5 and no 3Di-translation model.',
        'checked': 'PFMBench ranking confirmed. A remote-homology ROC-AUC figure of ~0.45 against '
                   '~0.06 sequence-only was read off a secondary page and NOT confirmed in the '
                   'article — do not quote it.',
    },
    'dplm2': {
        'model': 'DPLM-2 650M (multimodal diffusion protein LM)',
        'year': 2024, 'venue': 'ICLR 2025 (arXiv 2410.13782)',
        'paper': 'https://arxiv.org/pdf/2410.13782',
        'weights': 'https://huggingface.co/airkingbd/dplm2_650m',
        'embedding': 'One row per protein: mean over residues of the 1280-wide final hidden layer.',
        'code': 'NEW',
        'evidence': 'Thermostability Spearman: DPLM-2 650M 0.714, ESM-2 650M 0.691, sequence-only '
                    'DPLM 0.695. DeepLoc binary accuracy 93.64% against 92.28%. GO-CC Fmax 0.411 '
                    'against 0.402. From the same table, the negative side: DPLM-2 loses to '
                    'SaProt-650M on most of those tasks — SaProt thermostability 0.724, HumanPPI '
                    '86.41% against DPLM-2 84.44% — which the authors attribute to catastrophic '
                    'forgetting.',
        'source': 'https://arxiv.org/pdf/2410.13782',
        'distinct': 'Discrete diffusion over a joint sequence and structure-token vocabulary, so it is '
                    'trained by denoising rather than masking. Nothing in the pools is a diffusion model.',
        'checked': 'Confirmed including the negative side. Year field marked unconfirmed at 2025 — '
                   'arXiv is Oct 2024.',
    },
}

PROTEIN_CUT = {
    'saesm2': 'Two independent failures. The weights URL as given is dead, and a headline number '
              '(HumanPPI 92.68 against 91.96) does not appear in the source.',
    'venusplm': 'Evidence does not support it. Once a fabricated ordering is removed the whole case '
                'reduces to "ties ESM-2 650M", and ESM-2 is already a rep.',
}

# =============================================================================
# NUCLEOTIDE — nt_H3K27me3, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3,
#              nt_promoter_tata, gb_human_enhancers_cohn
# =============================================================================

NUCLEOTIDE = {
    'nucel': {
        'model': 'NucEL (single-nucleotide ELECTRA discriminator, 93M)',
        'year': 2025, 'venue': 'arXiv 2508.13191 / bioRxiv 2025.08.17.670700; AAAI 2026',
        'paper': 'https://arxiv.org/pdf/2508.13191',
        'weights': 'https://huggingface.co/FreakingPotato/NucEL — Apache-2.0',
        'embedding': 'One row per DNA window, 512 features. last_hidden_state is (1, L, 512); pool by '
                     'mean over tokens, or take the [CLS] token the paper itself uses.',
        'code': 'NEW',
        'evidence': 'Table 3, MCC, mean of 10 seeds, revised NT benchmark — five of these are the '
                    'exact datasets in the repo. H3K27me3: NucEL 0.611, DNABERT-2 0.599, '
                    'NT-HumanRef-500M 0.589, NT2-Multi-100M 0.595. H3K9me3: NucEL 0.474, DNABERT-2 '
                    '0.443, NT2-Multi 0.445. Promoter TATA: NucEL 0.922, DNABERT-2 0.784, '
                    'NT-HumanRef 0.831. NucEL LOSES two: H3K4me3 0.621 against DNABERT-2 0.646, '
                    'H3K9ac 0.530 against DNABERT-2 0.564.',
        'source': 'https://arxiv.org/pdf/2508.13191',
        'distinct': 'The only model here pretrained with replaced-token detection (ELECTRA '
                    'generator/discriminator) rather than masked-token or next-token prediction, and '
                    'single-nucleotide tokenised with hybrid local plus global attention.',
        'checked': 'Table 3 read directly; all five pairs confirmed including the two losses.',
        'flag': 'Best-matched evidence of any nucleotide candidate — four of its reported tasks ARE '
                'the repo\'s datasets, and dnabert2 and nt_v2_* are already in those pools, so both '
                'sides of its published comparison are inside the c13 subtraction.',
    },
    'generator_euk_1_2b': {
        'model': 'GENERator-eukaryote-1.2b-base',
        'year': 2025, 'venue': 'arXiv 2502.07272',
        'paper': 'https://arxiv.org/html/2502.07272v1',
        'weights': 'https://huggingface.co/GenerTeam/GENERator-eukaryote-1.2b-base — CC BY 4.0',
        'embedding': 'One row per window. Transformer decoder, 6-mer non-overlapping tokens, 98k bp '
                     'context; mean-pool last_hidden_state over tokens.',
        'code': 'NEW',
        'evidence': 'Table 4, Genomic Benchmarks accuracy, 10-fold CV — Human Enhancers Cohn: '
                    'GENERator 0.763, DNABERT-2 0.758, NT-v2 0.756, Caduceus-Ph 0.747, HyenaDNA '
                    '0.725. Coding vs Intergenomic 0.963 against NT-v2 0.955 and DNABERT-2 0.951. '
                    'Table 2, revised NT tasks, MCC — H3K27me3: GENERator 0.619, NT-v2 0.610, '
                    'DNABERT-2 0.599, Caduceus-Ph 0.547, HyenaDNA 0.541.',
        'source': 'https://arxiv.org/html/2502.07272v1',
        'distinct': 'Autoregressive decoder over 6-mer tokens at 98k bp context. Everything in the '
                    'nucleotide pools is either an encoder or a non-attention sequence model.',
        'checked': 'Both tables read directly; every value confirmed.',
        'flag': 'Reports on gb_human_enhancers_cohn and nt_H3K27me3 by name, against four models that '
                'are already in those pools.',
    },
    'omni_dna_116m': {
        'model': 'Omni-DNA 116M (family 20M-1B)',
        'year': 2025, 'venue': 'arXiv 2502.03499 (ICML 2025 submission)',
        'paper': 'https://arxiv.org/html/2502.03499v1',
        'weights': 'https://huggingface.co/zehui127/Omni-DNA-116M — CC BY 4.0',
        'embedding': 'One row per window. Autoregressive transformer, BPE vocabulary 4096, RoPE. '
                     'Hidden dimension 256-2048 across the family; mean-pool last_hidden_state.',
        'code': 'NEW',
        'evidence': 'Over the 18 NT downstream tasks: Omni-DNA-1B averages 0.767, Omni-DNA-116M 0.755, '
                    'DNABERT-2 0.679, NT-500M 0.698. Over Genomic Benchmarks (8 tasks): '
                    'Omni-DNA-116M 0.879, DNABERT-2 0.859, ranking first on five of eight.',
        'source': 'https://arxiv.org/html/2502.03499v1',
        'distinct': 'Pretrained next-token then expanded with task-specific tokens for multi-task '
                    'fine-tuning; the only candidate whose pretraining mixes DNA with text and image '
                    'target tokens. No nucleotide rep in the repo is cross-modal.',
        'checked': 'Confirmed on the arXiv HTML.',
    },
    'metagene_1': {
        'model': 'METAGENE-1 (7B metagenomic foundation model)',
        'year': 2025, 'venue': 'arXiv 2501.02045',
        'paper': 'https://arxiv.org/html/2501.02045v1',
        'weights': 'https://huggingface.co/metagene-ai/METAGENE-1 — CC BY 4.0',
        'embedding': 'One row per window, 4096 features. The paper\'s own embedding evaluations '
                     'mean-pool the last hidden state.',
        'code': 'NEW',
        'evidence': 'Gene-MTEB, mean-pooled FROZEN embeddings, no fine-tuning. Human-Virus '
                    'classification average: METAGENE-1 0.775, NT-2.5b-1000g 0.710, NT-2.5b-Multi '
                    '0.675, DNABERT-S 0.570, DNABERT-2 0.564. HMPD clustering average: 0.465 against '
                    '0.451 and 0.449.',
        'source': 'https://arxiv.org/html/2501.02045v1',
        'distinct': 'Pretrained on 1.5 trillion base pairs of wastewater metagenomic DNA and RNA, not '
                    'on reference genomes. Every nucleotide model in the repo is pretrained on '
                    'assembled reference genomes.',
        'checked': 'Gene-MTEB tables read directly.',
        'flag': 'Its evidence is on FROZEN mean-pooled embeddings, which is exactly what c13 builds. '
                'Most other nucleotide candidates report fine-tuned numbers instead.',
    },
    'evo2_1b': {
        'model': 'Evo 2 (1B base; 7B and 40B also released)',
        'year': 2025, 'venue': 'bioRxiv 2025.02.18.638918; Nature 2026',
        'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13128491/',
        'weights': 'https://huggingface.co/arcinstitute/evo2_1b_base — apache-2.0',
        'embedding': 'One row per window. StripedHyena 2, single-nucleotide tokens, 1 Mb context. '
                     'Per-token hidden states from a chosen layer, mean-pooled. Width depends on the '
                     'checkpoint.',
        'code': 'NEW',
        'evidence': 'Exon classification on held-out species, classifier fitted on Evo 2 7B '
                    'embeddings: AUROC 0.91-0.99 across eight species, beating Nucleotide '
                    'Transformer, Evo 1 and ab initio AUGUSTUS. Supervised BRCA1 variant '
                    'classification on Evo 2 embeddings: test AUROC 0.95, AUPRC 0.88, above zero-shot '
                    'Evo 2 40B on the same split.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13128491/',
        'distinct': 'StripedHyena 2 trained on 9.3 trillion nucleotides across all domains of life at '
                    '1 Mb context. The repo has hyenadna (much smaller Hyena) and caduceus (BiMamba) '
                    'but nothing at this scale or training breadth.',
        'checked': 'Read on the published PMC article.',
    },
    'janusdna': {
        'model': 'JanusDNA (Mamba-Attention Mixture-of-Experts, bidirectional)',
        'year': 2025, 'venue': 'NeurIPS 2025 (arXiv 2505.17257)',
        'paper': 'https://arxiv.org/html/2505.17257v1',
        'weights': 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FHDT0RN '
                   '— Apache-2.0 on the GitHub repo; the Dataverse weights state no separate license',
        'embedding': 'One row per window, 32 to 144 features depending on the configured sequence '
                     'length (0.4M to 7.7M activated parameters). Mean-pool the final hidden states. '
                     'By far the narrowest block on this list.',
        'code': 'NEW',
        'evidence': 'NT tasks, MCC — H3K4me3: JanusDNA 0.629 (no mid-attention) / 0.605 (with), '
                    'Caduceus-PH 0.544, HyenaDNA 0.549. H3K9ac: JanusDNA 0.641, Caduceus-PH 0.622, '
                    'HyenaDNA 0.581. Genomic Benchmarks, Human Enhancers Cohn accuracy: JanusDNA '
                    '0.745, Caduceus-PS 0.745 — a tie. The paper claims top performance on 12 of 18 '
                    'NT tasks.',
        'source': 'https://arxiv.org/html/2505.17257v1',
        'distinct': 'Combines autoregressive training with bidirectional reading through a '
                    'Mamba-Attention mixture-of-experts. caduceus is BiMamba without attention or '
                    'experts; mistral_dna is a plain decoder MoE.',
        'checked': 'Tables read directly, including the tie.',
        'flag': 'It wins the two NT tasks NucEL loses (H3K4me3, H3K9ac). Running both covers all four '
                'histone marks with a published claim on each.',
    },
    'moderngena_large': {
        'model': 'ModernGENA-large (377M) / ModernGENA-base (135M), AIRI',
        'year': 2026, 'venue': 'bioRxiv 2026.04.21.719816',
        'paper': 'https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full',
        'weights': 'https://huggingface.co/AIRI-Institute/moderngena-large — stated open-source; exact '
                   'license string not printed on the model card',
        'embedding': 'One row per window, 1024 features (large) or 768 (base). BPE vocabulary 32k over '
                     'A/T/G/C/N; mean-pool last_hidden_state. Loads with AutoModel and '
                     'trust_remote_code=True, needs flash_attention.',
        'code': 'NEW',
        'evidence': 'On the 18-task NT benchmark scored as 100xMCC, ModernGENA base and large both '
                    'take average rank 1 within the under-500M baseline set, ahead of GENA-LM at '
                    'rank 2, NTv2-250M at rank 3 and DNABERT-2 at rank 4. In the full table including '
                    'larger models, ModernGENA-large ranks 2nd overall behind GENERator-1.2B.',
        'source': 'https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1.full',
        'distinct': 'ModernBERT backbone (RoPE, alternating local/global attention, GeGLU, unpadding) '
                    'on 443 vertebrate assemblies, 353 Gbp. gena_lm is from the same lab but is the '
                    'older BERT-large T2T model trained on human only.',
        'checked': 'Ranks confirmed. WEAKNESS the second reader flagged: the paper reports RANKS, not '
                   'raw per-task MCC, so there is no raw number pair behind this entry.',
    },
    'borzoi': {
        'model': 'Borzoi (and Flashzoi)',
        'year': 2025, 'venue': 'Nature Genetics 2025; Flashzoi in Bioinformatics 41(9) btaf467',
        'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11985352/',
        'weights': 'https://github.com/johahi/borzoi-pytorch — MIT, four human replicate checkpoints',
        'embedding': 'One row per window. Emits 4,096 positions of width 1,536 at 128 bp resolution '
                     'before the species heads (stated in the Flashzoi paper), so mean-pooling a '
                     '200-1,000 bp window into one row is a slice, not new architecture.',
        'code': 'NEW',
        'evidence': 'Flashzoi beats Borzoi on eQTL effect Spearman at P = .01 and matches it on '
                    'prioritisation.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12457734/',
        'distinct': 'The only SUPERVISED sequence-to-function model on this list — everything else is '
                    'self-supervised, contrastive or distilled. Its pretraining heads include the '
                    'assay families H3K27me3, H3K4me3, H3K9ac and H3K9me3, which are the four histone '
                    'datasets.',
        'checked': 'Added by the completeness critic, not by the nucleotide lane. Weights, license and '
                   'the width/resolution figures confirmed; the Flashzoi comparison is the only '
                   'number pair.',
        'flag': 'EVALUATION OVERLAP. Its training targets ARE the labels being predicted on four of '
                'the six nucleotide datasets. A positive c13 delta from it would not be evidence the '
                'hybrid absorbed a new model — it would be evidence of label leakage through the '
                'pretraining set. Run it labelled that way or not at all.',
    },
    'plantcad2_large': {
        'model': 'PlantCAD2-Large (676M, Mamba2, 65 angiosperm genomes)',
        'year': 2025, 'venue': 'bioRxiv 2025.08.27.672609; Cell Genomics 2026',
        'paper': 'https://www.biorxiv.org/content/10.1101/2025.08.27.672609v1',
        'weights': 'https://huggingface.co/kuleshov-group/PlantCAD2-Large-l48-d1536 — license not '
                   'stated on the collection page read',
        'embedding': 'One row per window, 1536 features (d1536 in the checkpoint name).',
        'code': 'NEW',
        'evidence': 'Abstract, verbatim: surpasses the 7B Evo2 model in 10 of 12 tasks; outperforms '
                    'the 1B AgroNT across seven cross-species tasks including chromatin accessible '
                    'region, gene expression and protein translation; maize accessible chromatin '
                    'AUPRC 0.587 to 0.711.',
        'source': 'https://www.biorxiv.org/content/10.1101/2025.08.27.672609v1',
        'distinct': 'Mamba2 over 65 angiosperm genomes.',
        'checked': 'All three claims confirmed verbatim in the abstract. WEAKNESS: abstract-only '
                   'self-report; neither Evo2\'s nor AgroNT\'s numbers are printed, and the '
                   '0.587-to-0.711 pair is PlantCAD2 against ITSELF at a shorter context, not against '
                   'a competitor.',
        'flag': 'Every number is on plant tasks. The six nucleotide datasets are human. Kept on the '
                'list because "wrong organism in the evidence" is not one of the five admission '
                'rules — cut it yourself if you want the evidence to be on-task.',
    },
}

NUCLEOTIDE_CUT = {
    'generator_v2_1_2b': 'No comparator number exists anywhere in the paper. ClinVar AUPRC 0.934 '
                         '(v2-1B) / 0.942 (v2-3B) stands alone; Table 2 compares only against '
                         'GENERator-v1; the NT series and HyenaDNA are dismissed as "substantially '
                         'lower performance" with no printed values; the Evo2 line names a comparator '
                         'without its number. Fails admission rule 4. GENERator-v1 is on the list '
                         'with a full per-task table, same backbone, same 6-mer tokenizer, same 1.2B '
                         'size, so v2 costs a second search pass and adds no measurable claim.',
    'hybridna_300m': 'Weights are a third-party upload under a personal account, not the authors\'.',
    'alphagenome': 'Weights gated behind non-commercial terms, and no embedding path is exposed.',
}

# =============================================================================
# REACTION — buchwald_hartwig, bh_doyle_plates
# =============================================================================

REACTION = {
    'rxngraphormer': {
        'model': 'RXNGraphormer (RXNEmb reaction embedding)',
        'year': 2025, 'venue': 'Nature Machine Intelligence 7(9) 1561-1571, DOI 10.1038/s42256-025-01098-4',
        'paper': 'https://www.researchsquare.com/article/rs-5994908/v1.pdf',
        'weights': 'https://doi.org/10.6084/m9.figshare.28356077 — MIT',
        'embedding': 'One 768-length vector per reaction. Atom node states from the graph encoder are '
                     'pooled by a global attention layer that weights atoms, then the reactant-side '
                     'and product-side transformer states are combined.',
        'code': 'NEW',
        'evidence': 'Doyle Buchwald-Hartwig, random split: R2 0.971, MAE 2.98 yield-percentage points '
                    'over ten trials. Head to head on the same set: MAE 3.02 against ReaMVP 3.11. On '
                    'the seven out-of-sample splits (three aryl-halide, four additive) it beats '
                    'MFF+RF, YieldBERT and GraphRXN on every one.',
        'source': 'https://www.researchsquare.com/article/rs-5994908/v1.pdf',
        'distinct': 'Pretrained on 13 million reactions (6.8M real plus an equal number of fictitious '
                    'bond-change reactions), GNN for intramolecular and transformer for '
                    'intermolecular. Nothing in either reaction pool has that pretraining.',
        'checked': 'Year, weights and numbers confirmed.',
        'flag': 'Its own follow-up says the frozen embedding "faces challenges" on catalyst- and '
                'ligand-varying high-throughput sets — which is what both reaction datasets are.',
    },
    'reamvp': {
        'model': 'ReaMVP',
        'year': 2024, 'venue': 'Journal of Cheminformatics 16, DOI 10.1186/s13321-024-00815-2',
        'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10895839/',
        'weights': 'https://github.com/Meteor-han/ReaMVP/tree/main/checkpoint/rnn_geo — MIT',
        'embedding': 'One vector per reaction: a 256-dimension bidirectional-GRU sequence encoder '
                     'concatenated with a SchNet 3D-conformer encoder; hidden layers 128 wide.',
        'code': 'NEW',
        'evidence': 'Doyle Buchwald-Hartwig, random split: R2 0.971 +/- 0.002, RMSE 4.626 +/- 0.139 '
                    'yield-percentage points. Four additive out-of-sample splits: R2 0.844, 0.896, '
                    '0.792, 0.693. Independently, RXNGraphormer scores ReaMVP at MAE 3.11 against its '
                    'own 3.02; HiCLR scores ReaMVP at R2 0.212 against its own 0.248 on the '
                    'Buchwald-Hartwig notebook set.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10895839/',
        'distinct': 'Two-stage pretraining aligning a 1D SMILES view with a 3D conformer view of the '
                    'same reaction. The only candidate whose reaction vector carries 3D geometry; '
                    'neither reaction pool has a 3D reaction rep.',
        'checked': 'Confirmed, including both independent third-party scores.',
    },
    'egret': {
        'model': 'Egret',
        'year': 2024, 'venue': 'Research (AAAS), PMC10777739',
        'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10777739/',
        'weights': 'https://drive.google.com/file/d/1MNzGyMYtNDt-TshKtsJkEEjHGtXTJhLN/view — MIT',
        'embedding': 'One 256-length vector per reaction. BERT encoder, 12 layers, 4 heads, hidden '
                     'size 256, pooled over reaction-SMILES tokens the same way rxnfp is pooled.',
        'code': 'NEW',
        'evidence': 'Doyle additive out-of-sample splits, R2 — Test2: Egret 0.88 +/- 0.03, Yield-BERT '
                    '0.84 +/- 0.03, DRFP 0.83 +/- 0.003. Test4: 0.54 +/- 0.06 against 0.49 +/- 0.05 '
                    'and 0.49 +/- 0.004. Test1: 0.84 +/- 0.01, level with Yield-BERT. It LOSES Test3: '
                    '0.65 +/- 0.06 against Yield-BERT 0.75 +/- 0.04.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10777739/',
        'distinct': 'Pretrained on Pistachio with masked language modelling plus a contrastive loss '
                    'over reaction CONDITIONS, so two reactions with the same reactants but different '
                    'solvent or catalyst are pushed apart. rxnfp, the nearest thing in the pools, has '
                    'no condition objective.',
        'checked': 'Confirmed including the loss on Test3.',
        'flag': 'CRITICAL. The completeness critic found the checkpoint at that Drive link is the '
                'Buchwald-Hartwig-FINETUNED yield head. Embedding buchwald_hartwig or bh_doyle_plates '
                'with it is training on the labels. Find the pretrained-only checkpoint before this '
                'is runnable, or its c13 delta is meaningless.',
    },
    'unimol3': {
        'model': 'Uni-Mol3',
        'year': 2025, 'venue': 'arXiv 2508.00920',
        'paper': 'https://arxiv.org/html/2508.00920v2',
        'weights': 'https://github.com/LirongWu/Uni-Mol3 — MIT (repo). Reaction pre-training '
                   'checkpoint on Google Drive; Uni-Mol2 84M base at https://huggingface.co/dptech/Uni-Mol2',
        'embedding': 'Hidden dimension 768. T5 backbone, 8 heads, 8 encoder and 8 decoder layers, '
                     'initialised from Uni-Mol2 84M. A molecule becomes discrete 3D-aware tokens via '
                     'its Mol-Tokenizer, then a reaction is the token sequence.',
        'code': 'NEW',
        'evidence': 'Doyle additive out-of-sample splits, R2 — Test1: Uni-Mol3 0.874, YieldBert 0.838, '
                    'DRFP 0.810, Chemprop 0.798. Test2: 0.909 against YieldBert 0.842 and T5Chem '
                    '0.895. Test3: 0.769 against T5Chem 0.765. It LOSES Test4: 0.525 against T5Chem '
                    '0.610.',
        'source': 'https://arxiv.org/html/2508.00920v2',
        'distinct': 'The only candidate whose tokens encode 3D molecular structure discretely, and the '
                    'only one pretrained in two stages (molecule grammar, then reaction).',
        'checked': 'Confirmed including the Test4 loss.',
        'flag': 'The critic found it ships no embedding-extraction entry point. A pooling hook has to '
                'be written against the encoder.',
    },
    'diff_chemeleon': {
        'model': 'CheMeleon, applied through the per-component construction the reaction pools '
                 'already use. `concat_chemeleon` is the same entry with the other slot rule and '
                 'is registered alongside it.',
        'year': 2025, 'venue': 'arXiv 2506.15792',
        'paper': 'https://arxiv.org/abs/2506.15792',
        'weights': 'https://doi.org/10.5281/zenodo.15426600 — MIT code, CC0 weights',
        'embedding': 'One row per reaction, 2048 features per component slot. Built the same way '
                     'diff_ecfp4, concat_ecfp4, diff_maccs, concat_maccs and component_descriptors '
                     'are built: embed each component, then concatenate or difference in fixed slot '
                     'order.',
        'code': 'NEW builder for CheMeleon; the reaction wrapper already exists as '
                'create_reaction_rep_variant in _shared._rxn_generators.',
        'evidence': 'Same as the molecular entry: 75% win rate over 28 Polaris tasks against Random '
                    'Forest 68%, fastprop 36%, Chemprop 32%; 97% over 30 MoleculeACE assays against '
                    'Random Forest 50%.',
        'source': 'https://arxiv.org/abs/2506.15792',
        'distinct': 'No new architecture at all. Five of the ten representations in each reaction pool '
                    'are already built by this construction, so this substitutes a 2025 encoder into '
                    'a slot the pipeline already fills with ECFP and MACCS.',
        'checked': 'Added by the completeness critic. The construction was verified against the repo: '
                   'component_descriptors, concat_ecfp4, concat_maccs, diff_ecfp4 and diff_maccs all '
                   'build a reaction row this way, and five of those sit in the two pools today.',
        'flag': 'The critic notes the same applies to MolE and Uni-Mol2. This is the cheapest route '
                'to a genuinely new reaction feature block, because reaction-native encoders barely exist.',
    },
}

REACTION_CUT = {
    'hiclr': 'PENDING RECHECK, not cut on substance. The Zenodo record exists and DataCite reports it '
             'findable under CC BY 4.0, but Zenodo was unreachable during the check, so no one has '
             'confirmed checkpoint.zip is in it. The repo carries no license file and the paper has '
             'no availability statement pointing at those weights. Re-run the weights check; if the '
             'file is there it clears.',
    'smi_ted': 'Headline out-of-sample number is not believable. SMI-TED reports the four Doyle '
               'additive splits at 0.9832 / 0.9820 / 0.9827 / 0.9825 — flat across all four, and '
               'statistically indistinguishable from its own random-split score of 0.9841. Every '
               'other source shows those splits at wildly different difficulty with Tests 3 and 4 '
               'collapsing: Egret 0.84 / 0.88 / 0.65 / 0.54, DRFP 0.81 / 0.83 / 0.71 / 0.49, '
               'Uni-Mol3 0.874 / 0.909 / 0.769 / 0.525. An out-of-sample split that scores the same '
               'as a random split is the signature of a split that is not out-of-sample. The paper '
               'gives no split definition. NOTE: smited IS in five molecular pools as a molecular '
               'rep — this cut is about the reaction claim only.',
    'moldescpred': 'Headline number does not hold. It compares only against YieldMPNN and calls it a tie.',
}

# =============================================================================
# OMICS / TRANSCRIPTOMICS — tcga_luad_vs_lusc, tcga_brca_pam50
# =============================================================================

OMICS = {
    'bulkformer': {
        'model': 'BulkFormer (147M)',
        'year': 2025, 'venue': 'Cell Systems 2026 (bioRxiv 2025-06-11)',
        'paper': 'https://www.biorxiv.org/content/10.1101/2025.06.11.659222v1.full',
        'weights': 'https://github.com/KangBoming/BulkFormer — MIT, five checkpoints 37M-147M via Google Drive',
        'embedding': 'One row per bulk sample. The encoder emits a contextual embedding for each of '
                     '20,010 protein-coding genes; a sample-level MLP head compresses the global '
                     'expression vector to one fixed vector.',
        'code': 'NEW',
        'evidence': 'Ten-fold CV disease/phenotype classification weighted F1: BulkFormer 0.939, '
                    'scGPT 0.885 (next best). Cancer subtype classification weighted F1 0.833. '
                    'Pan-cancer prognosis across 33 cancer types AUROC 0.747 against scFoundation '
                    '0.726. Transcriptome imputation r = 0.954.',
        'source': 'https://www.biorxiv.org/content/10.1101/2025.06.11.659222v1.full',
        'distinct': 'The only model here pretrained on BULK RNA-seq itself: 581,503 human bulk '
                    'profiles, ~20,000 genes per sample, no single-cell sparsity assumption. The '
                    'repo\'s eight existing foundation-model reps are all single-cell-pretrained '
                    'except bulkrnabert.',
        'checked': 'Numbers confirmed. Year field marked unconfirmed at 2026 — the preprint is Jun 2025.',
        'flag': 'Best-matched candidate in this domain: bulk input, and its evidence task is cancer '
                'subtype classification, which is what tcga_brca_pam50 is. Its comparators scGPT and '
                'scFoundation are both already reps.',
    },
    'transcriptformer': {
        'model': 'TranscriptFormer (TF-Sapiens / TF-Metazoa / TF-Exemplar, CZI)',
        'year': 2025, 'venue': 'bioRxiv 2025-04-25',
        'paper': 'https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1.full',
        'weights': 'https://github.com/czi-ai/transcriptformer — license not stated on the pages read',
        'embedding': 'One row per sample, 2048 features. Mean over the final-layer contextual gene '
                     'embeddings of all expressed genes, excluding the assay token.',
        'code': 'NEW',
        'evidence': 'Cell type classification on Tabula Sapiens 2.0 (26 tissues), macro F1: TF-Metazoa '
                    '0.910 +/- 0.001, UCE 0.906 +/- 0.001. Disease state prediction on a COVID-19 '
                    'dataset, macro F1: TF-Sapiens 0.859, UCE 0.805, scGPT 0.798, Geneformer 0.797. '
                    'Cross-species spermatogenesis transfer, average F1: TF-Exemplar 0.480 +/- 0.002, '
                    'UCE 0.377 +/- 0.002.',
        'source': 'https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1.full',
        'distinct': 'Generative rather than masked-token pretraining, 112 million cells across 12 '
                    'species, with an explicit assay token.',
        'checked': 'Confirmed. The disease-state number is the closest published proxy to a TCGA '
                   'tumour classification on this list, and its three comparators (UCE, scGPT, '
                   'Geneformer) are all already reps.',
    },
    'cellfm': {
        'model': 'CellFM (80M)',
        'year': 2025, 'venue': 'Nature Communications 2025',
        'paper': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12092794/',
        'weights': 'https://huggingface.co/ShangguanNingyuan/CellFM — CC BY-NC-ND 4.0',
        'embedding': 'One row per sample, 1536 features. Input capped at 2048 genes: beyond that the '
                     'model takes 2048 sampled from the high-expression genes, so a bulk sample '
                     'enters as its top-expressed genes.',
        'code': 'NEW',
        'evidence': 'Cell type annotation accuracy, intra-dataset, averaged over eight datasets: '
                    'CellFM 92.91%, scGPT 90.94%, scFoundation 90.89%, Geneformer 82.71%, scBERT 82.71%.',
        'source': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12092794/',
        'distinct': 'Pretrained on 100 million human cells, the largest human-only corpus here, with a '
                    '24,078-gene vocabulary. Beats scgpt and scfoundation, both already reps, on the '
                    'same benchmark.',
        'checked': 'The accuracy table confirmed. The second reader marked the evidence field '
                   'not-fully-confirmed and flagged the reference implementation as a practical caveat.',
        'flag': 'Non-commercial no-derivatives license. The 2048-gene cap means a bulk TCGA vector is '
                'truncated on the way in.',
    },
    'scprint2': {
        'model': 'scPRINT-2',
        'year': 2025, 'venue': 'bioRxiv 2025-12-11',
        'paper': 'https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1.full',
        'weights': 'stated in the paper: pretrained weights plus all 42 ablation checkpoints — GPL-v3',
        'embedding': 'One row per sample, 256 features (base) or 512 (medium), from the '
                     'encoder-compressor bottleneck.',
        'code': 'NEW',
        'evidence': 'Open Problems benchmark, November 2025, zero-shot average: scPRINT-2 75%, '
                    'scPRINT-1 47%, other zero-shot foundation models 40-60%. Cell embedding quality, '
                    'cross-organism scIB: scPRINT-2 zero-shot 0.49, PCA baseline 0.37, rising to 0.60 '
                    'with fine-tuning.',
        'source': 'https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1.full',
        'distinct': '20M active parameters with an encoder-compressor-decoder design, so the embedding '
                    'is a genuine low-dimensional bottleneck rather than a pooled token matrix. The '
                    'only candidate whose paper prints a PCA baseline next to its own number — and '
                    'gene_pca100 is in both TCGA pools.',
        'checked': 'Confirmed. The second reader marked the evidence field not-fully-confirmed.',
    },
    'aido_cell': {
        'model': 'AIDO.Cell-100M (also published as GB.Cell-100M)',
        'year': 2024, 'venue': 'bioRxiv 2024-11-28; NeurIPS 2024 Workshop on AI for New Drug Modalities',
        'paper': 'https://virtualcellmodels.cziscience.com/model/gb-cell',
        'weights': 'https://huggingface.co/genbio-ai/AIDO.Cell-100M — GenBio AI Community License '
                   '(academic/non-commercial with attribution)',
        'embedding': 'One row per sample. 18 layers, hidden size 650, 20 heads; context is the entire '
                     '19,264-gene transcriptome with NO truncation, so a bulk vector is fed whole and '
                     'the gene-token outputs are mean-pooled.',
        'code': 'NEW',
        'evidence': 'Cell type classification macro F1: AIDO.Cell-100M 0.761 on Zheng68K and 0.910 on '
                    'Segerstolpe.',
        'source': 'https://virtualcellmodels.cziscience.com/model/gb-cell',
        'distinct': 'Full 19,264-gene context with a read-depth-aware objective and auto-discretised '
                    'continuous expression values — the closest architectural match to a dense bulk '
                    'expression vector of anything here apart from BulkFormer.',
        'checked': 'The two F1 values confirmed. WEAKNESS: comparators are NAMED on the CZI page '
                   '(scFoundation-100M, scBERT, CellTypist, scANVI, ACTINN, Scanpy, SingleCellNet) '
                   'but neither the model page nor the bioRxiv HTML prints their values, so this is a '
                   'number without a matched comparator number.',
    },
    'methyl_methylgpt': {
        'model': 'MethylGPT (base / medium / large)',
        'year': 2024, 'venue': 'bioRxiv 2024-10-30 v2',
        'paper': 'https://www.biorxiv.org/content/10.1101/2024.10.30.621013v2.full',
        'weights': 'https://github.com/albert-ying/MethylGPT — Apache 2.0 on the code, three '
                   'checkpoints via Google Drive; the weight files\' terms are not stated separately',
        'embedding': 'One row per sample, 64 features, from a [CLS] token prepended to the CpG '
                     'sequence. Input is 49,156 curated CpG sites, which the TCGA 450K arrays cover.',
        'code': 'NEW',
        'evidence': 'Disease prediction across 60 conditions on Generation Scotland (n=18,859): AUC '
                    '0.736 validation, 0.720 test. Age prediction median absolute error 4.45 years. '
                    'Methylation value prediction MAE 0.074, Pearson 0.929.',
        'source': 'https://www.biorxiv.org/content/10.1101/2024.10.30.621013v2.full',
        'distinct': 'A METHYLATION block, not an expression block. The repo\'s methylation features '
                    'are methyl_pca50 and methyl_hvg, both unsupervised reductions of the raw beta '
                    'matrix. A pretrained methylome encoder is a different kind of feature entirely, '
                    'and both TCGA pools already carry methylation.',
        'checked': 'Confirmed. ElasticNet, AltumAge and Horvath\'s clock are named as beaten on age '
                   'prediction but their MedAE values are NOT printed on the page read.',
        'flag': 'Only 64 features. That is the narrowest block on the whole list, which matters for '
                'how a budget gets split across it.',
    },
    'methyl_cpgpt': {
        'model': 'CpGPT',
        'year': 2024, 'venue': 'bioRxiv 2024-10-24',
        'paper': 'https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1.full',
        'weights': 'https://huggingface.co/lucascamillomd/cpgpt-models — MIT, with a carve-out '
                   'forbidding use in biomarkers-of-aging prize competitions',
        'embedding': 'One row per sample, from a learnable classification token aggregating across the '
                     'CpG sequence. The dimension varies by checkpoint and is not stated on the '
                     'preprint page read.',
        'code': 'NEW',
        'evidence': 'Mortality prediction on a held-out test cohort: C-index 0.82, AUC 0.90 (training '
                    'cohorts 0.68-0.82). Reconstruction of a full 450K array from the Hannum subset: '
                    'median Pearson 0.990, median Spearman 0.964, median MAE 0.028.',
        'source': 'https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1.full',
        'distinct': 'A second, independent methylome encoder, pretrained on 150,000+ samples from '
                    '2,000+ studies with cross-platform reconstruction built in. Running it alongside '
                    'MethylGPT is what separates "the methylation modality carries something" from '
                    '"this one methylation model carries something".',
        'checked': 'Confirmed. WEAKNESS: the only named comparator is mLiftOver for platform '
                   'conversion and no number is given for it, so these are numbers without a matched '
                   'comparator number.',
    },
    'tahoe_x1': {
        'model': 'Tahoe-x1 (70M / 1.3B / 3B)',
        'year': 2025, 'venue': 'bioRxiv 2025-10-23 (Tahoe Therapeutics)',
        'paper': 'https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1.full',
        'weights': 'https://huggingface.co/tahoebio/Tahoe-x1 — Apache 2.0',
        'embedding': 'One row per sample, written to adata.obsm by predict_embeddings.py. The model '
                     'card does not print the dimension.',
        'code': 'NEW',
        'evidence': 'MSigDB hallmark pathway membership recovery, AUPRC over five seeds: Tahoe-x1-3B '
                    '0.31, ahead of SE-600M, TranscriptFormer (Metazoa) and nv-Geneformer-100M. Cell '
                    'type classification on Tabula Sapiens 2.0: Tx1-3B matches or exceeds '
                    'TranscriptFormer (Metazoa) despite human-only training.',
        'source': 'https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1.full',
        'distinct': 'Pretrained on 266 million cells including the Tahoe-100M drug perturbation '
                    'compendium — the only cancer-perturbation-trained encoder here. Nothing in the '
                    'pool has seen chemical perturbation.',
        'checked': 'Confirmed. The completeness critic marks this the WEAKEST omics entry: one '
                   'self-reported AUPRC with three named comparators and ZERO comparator values, and '
                   'no stated embedding width, so the block cannot be costed against a budget.',
    },
    'c2s_scale': {
        'model': 'C2S-Scale (Gemma-2, 1B / 2B / 9B / 27B; Yale van Dijk lab, Google Research/DeepMind)',
        'year': 2025, 'venue': 'bioRxiv 2025-04-14',
        'paper': 'https://www.biorxiv.org/content/10.1101/2025.04.14.648850v2.full',
        'weights': 'https://huggingface.co/vandijklab/C2S-Scale-Gemma-2-27B and the '
                   'vandijklab/cell2sentence-models collection — CC-BY-4.0',
        'embedding': 'One row per sample: mean-pooled final hidden states over the "cell sentence" — '
                     'the expression profile rewritten as a rank-ordered list of gene names. '
                     'Dimension is the Gemma-2 hidden size of the chosen checkpoint.',
        'code': 'NEW',
        'evidence': 'Cell type annotation on a diverse immune tissue dataset: C2S-Scale 95.43%, '
                    'scGPT 93.1%, Geneformer 94.0%.',
        'source': 'https://www.biorxiv.org/content/10.1101/2025.04.14.648850v2.full',
        'distinct': 'The only text-LLM route here: a gene profile becomes a rank-ordered sentence of '
                    'gene symbols and a general-purpose LLM encodes it. Its inductive bias has '
                    'nothing in common with the masked-expression encoders in the pool.',
        'checked': 'CAVEAT STATED BY THE READER: bioRxiv rate-limited every attempt to open that page, '
                   'so those three numbers were read off the search index of the bioRxiv URL, not off '
                   'the page. Verify before building.',
        'flag': 'The completeness critic demotes this: 95.43% measures the accuracy of generated text '
                'naming a cell type, NOT the quality of a mean-pooled hidden state. The number does '
                'not describe the object c13 would build, and it costs a 27B forward pass per sample.',
    },
}

OMICS_CUT = {
    'sclong': 'Weights not publicly hosted. A 1B-parameter checkpoint whose sole download is one '
              'graduate student\'s institutional OneDrive.',
    'genebag': 'No pretrained encoder weights exist publicly — what is deposited is six task-specific '
               'heads.',
}

# =============================================================================
# PATHOLOGY / WSI — tcga_luad_vs_lusc, tcga_brca_pam50
# =============================================================================
# Read the critic note at the bottom of this block before adding anything here.

PATHOLOGY_WSI = {
    'wsi_gpfm': {
        'model': 'GPFM (Generalizable Pathology Foundation Model, unified knowledge distillation)',
        'year': 2024, 'venue': 'Nature Biomedical Engineering s41551-025-01488-4, online 2 Sept 2025; '
                               'preprint arXiv 2407.18449',
        'paper': 'https://github.com/birkhoffkiki/GPFM/',
        'weights': 'https://huggingface.co/majiabo/GPFM — license not stated on the repository page read',
        'embedding': 'Tile encoder. One tile row = 1024 features per the repo\'s own extraction code. '
                     'One slide row = mean-pool tile vectors over the slide.',
        'code': 'NEW',
        'evidence': 'Across 36 whole-slide-image classification tasks: GPFM average rank 1.22 against '
                    'UNI at 3.60, mean AUC 0.891, ahead of UNI by 1.6 percentage points. Across the '
                    'full 72-task suite: GPFM average rank 1.6 with 42 first places, UNI 3.7 with 6. '
                    'An independent 2026 distillation paper puts GPFM at 0.789 mean on the seven-task '
                    'EVA benchmark against Virchow2 0.810 and UNI2.',
        'source': 'https://github.com/birkhoffkiki/GPFM/',
        'distinct': 'Pretrained by distilling several expert teachers plus self-distillation, not by '
                    'plain DINOv2 self-supervision like every model currently in the WSI pool. 1024 '
                    'features, a width no existing WSI entry uses.',
        'checked': 'Confirmed, including the independent EVA number that puts it BELOW Virchow2. '
                   'Year field marked unconfirmed at 2025 — the preprint is Jul 2024.',
    },
    'wsi_midnight': {
        'model': 'Midnight-12k (kaiko.ai, ViT-g/14)',
        'year': 2025, 'venue': 'MICCAI 2025 (arXiv 2504.05186)',
        'paper': 'https://arxiv.org/abs/2504.05186',
        'weights': 'https://huggingface.co/kaiko-ai/midnight — MIT for Midnight-12k. The Midnight-92k '
                   'variants are restricted.',
        'embedding': 'Tile encoder. One tile row = the class token concatenated with the mean of the '
                     'patch tokens, as the model card instructs. One slide row = mean-pool tiles.',
        'code': 'NEW',
        'evidence': 'Table 2, average over 14 evaluation tasks: Midnight-12k 0.763, Midnight-92k/392 '
                    '0.778, UNI-2 0.776, Virchow2 0.766, Kaiko-B8 0.757, H-optimus-0 0.755, '
                    'Prov-GigaPath 0.752, Hibou-L 0.751, Phikon-v2 0.718. Per-task: PCam 0.803, '
                    'BACH 0.907, CRC 0.967, Camelyon16 0.869.',
        'source': 'https://arxiv.org/abs/2504.05186',
        'distinct': 'MIT-licensed and ungated, which almost nothing else in this modality is. Its '
                    'table scores UNI-2, Virchow2, H-optimus-0, Prov-GigaPath and Phikon-v2 — five '
                    'models the repo already holds or has keyed.',
        'checked': 'Table 2 confirmed. The paper states its own evaluation-overlap caveat.',
    },
    'wsi_distillpath': {
        'model': 'DistillPath (KS16-Virchow2 / KS16-H0mini)',
        'year': 2026, 'venue': 'arXiv 2608.17872, ECCV 2026 MedFM-Bench workshop',
        'paper': 'https://arxiv.org/abs/2608.17872',
        'weights': 'see the paper',
        'embedding': 'Tile encoder, output dimension 384 (confirmed in the methods). One slide row = '
                     'mean-pool tiles.',
        'code': 'NEW',
        'evidence': 'EVA seven-task mean: DistillPath-KS16-Virchow2 0.795, Virchow2 0.810, UNI2-h '
                    '0.806, H-optimus-0 0.803, GPFM 0.789. HEST mean Pearson r: '
                    'DistillPath-KS16-H0mini 0.387, Virchow2 0.398.',
        'source': 'https://arxiv.org/abs/2608.17872',
        'distinct': 'A distilled 384-wide encoder — by far the narrowest WSI block, against 1024-1536 '
                    'for everything in the pool.',
        'checked': 'Every number confirmed. This is the model\'s OWN paper against published '
                   'comparator scores; no independent benchmark of it exists yet. Note it loses to '
                   'Virchow2 on both metrics.',
    },
    'wsi_moozy': {
        'model': 'MOOZY (patient-first foundation model, Atlas Analytics Lab)',
        'year': 2026, 'venue': 'ECCV 2026',
        'paper': 'https://github.com/AtlasAnalyticsLab/MOOZY',
        'weights': 'https://huggingface.co/AtlasAnalyticsLab/MOOZY — license not stated on the page read',
        'embedding': 'CASE-level encoder. One PATIENT row = 768 features, from a learnable [CASE] '
                     'token in a case transformer attending over all that patient\'s slide embeddings.',
        'code': 'NEW',
        'evidence': 'Sixteen held-out tasks, five-fold MLP probe, macro averages: MOOZY weighted F1 '
                    '0.769 and balanced accuracy 0.702, against TITAN 0.758 / 0.683, CHIEF 0.740 / '
                    '0.668, Madeleine 0.751 / 0.671, PRISM 0.738 / 0.656, GigaPath 0.735 / 0.649. '
                    'TITAN takes weighted ROC-AUC 0.768 against MOOZY 0.763, so the win is not clean '
                    'on every metric.',
        'source': 'https://github.com/AtlasAnalyticsLab/MOOZY',
        'distinct': 'The only candidate whose unit of representation is the PATIENT rather than the '
                    'slide, and it beats TITAN, which the repo already has. Weights are ungated.',
        'checked': 'Confirmed against the repository.',
        'flag': 'The completeness critic demotes this: every number is from the model\'s own README, '
                'and it is pretrained on 77,134 public slides, which very likely includes the TCGA '
                'slides both target datasets evaluate on. That overlap is not stated anywhere, unlike '
                'Midnight-12k\'s.',
    },

    # ── Wired in the repo, in neither TCGA pool ──────────────────────────────
    # These three are keyed at tests/find_best_hybrid_tcga.py:280-282 and
    # dimensioned at src/kirby/datasets/tcga.py:163-165, and neither
    # tcga_luad_vs_lusc nor tcga_brca_pam50 holds any of them. They cost feature
    # extraction, not a builder, which makes them the cheapest candidates in the
    # whole list. They are on it for that reason, and the evidence field says
    # exactly how much published backing each one has.
    'wsi_virchow2': {
        'model': 'Virchow2 (Paige AI)',
        'year': 2024, 'venue': 'Paige AI',
        'paper': 'https://arxiv.org/abs/2504.05186 (Midnight, where it is benchmarked)',
        'weights': 'gated on HuggingFace; the repo already extracts its features',
        'embedding': 'Tile encoder, 1280 features per tile (tcga.py:163). One slide row = '
                     'mean-pool tiles.',
        'code': "KEYED, NOT POOLED. 'wsi_virchow2' at find_best_hybrid_tcga.py:280, dimension at "
                'tcga.py:163. In neither TCGA pool.',
        'evidence': 'Two independent tables. Midnight Table 2, average over 14 tasks: Virchow2 '
                    '0.766, UNI-2 0.776, Midnight-12k 0.763, H-optimus-0 0.755, Prov-GigaPath '
                    '0.752, Phikon-v2 0.718. DistillPath EVA seven-task mean: Virchow2 0.810, '
                    'UNI2-h 0.806, H-optimus-0 0.803, DistillPath 0.795, GPFM 0.789.',
        'source': 'https://arxiv.org/abs/2504.05186 and https://arxiv.org/abs/2608.17872',
        'distinct': 'Tops the EVA seven-task benchmark among everything checked in this '
                    'compilation, and the two TCGA pools hold titan, ctranspath, hoptimus0, '
                    'gigapath and phikon but not this.',
        'checked': 'Both tables read directly by the second reader. It is the best-evidenced '
                   'pathology entry here and the only one that wins its benchmark outright.',
    },
    'wsi_hoptimus1': {
        'model': 'H-optimus-1 (Bioptimus)',
        'year': 2025, 'venue': 'Bioptimus',
        'paper': 'https://huggingface.co/bioptimus',
        'weights': 'https://huggingface.co/bioptimus; the repo already extracts its features',
        'embedding': 'Tile encoder, 1536 features per tile (tcga.py:165). One slide row = '
                     'mean-pool tiles.',
        'code': "KEYED, NOT POOLED. 'wsi_hoptimus1' at find_best_hybrid_tcga.py:282, dimension at "
                'tcga.py:165. In neither TCGA pool.',
        'evidence': 'NO CONFIRMED COMPARATOR PAIR. Its predecessor H-optimus-0 is in the '
                    'tcga_brca_pam50 pool and scores 0.755 over Midnight\'s 14 tasks against '
                    'UNI-2 0.776 and Virchow2 0.766, but no reader in this compilation opened a '
                    'page giving H-optimus-1 itself a number with a named comparator and that '
                    'comparator\'s number.',
        'source': 'https://arxiv.org/abs/2504.05186 (for H-optimus-0 only)',
        'distinct': 'The successor to a model already in one TCGA pool, which makes its c13 delta '
                    'a version-to-version comparison rather than a new-family one.',
        'checked': 'The repo wiring was confirmed by reading find_best_hybrid_tcga.py and tcga.py. '
                   'It is on this list because it is already wired, NOT because admission rule 4 '
                   'was met — it was not. Treat its result as exploratory.',
    },
    'wsi_conch': {
        'model': 'CONCH v1.5 (Mahmood Lab)',
        'year': 2024, 'venue': 'Mahmood Lab',
        'paper': 'https://huggingface.co/MahmoodLab',
        'weights': 'gated on HuggingFace; the repo already extracts its features',
        'embedding': 'Vision-language tile encoder, 768 features per tile (tcga.py:164). One slide '
                     'row = mean-pool tiles.',
        'code': "KEYED, NOT POOLED. 'wsi_conch' at find_best_hybrid_tcga.py:281, dimension at "
                'tcga.py:164. In neither TCGA pool.',
        'evidence': 'NO CONFIRMED COMPARATOR PAIR. No reader in this compilation opened a page '
                    'giving CONCH a number with a named comparator and that comparator\'s number.',
        'source': 'https://huggingface.co/MahmoodLab',
        'distinct': 'The only vision-LANGUAGE tile encoder among the seven WSI reps the repo '
                    'holds — everything in the pools is image-only self-supervision. Its 768 '
                    'features are also the narrowest of the wired WSI reps.',
        'checked': 'The repo wiring was confirmed by reading find_best_hybrid_tcga.py and tcga.py. '
                   'On this list because it is already wired, NOT because admission rule 4 was '
                   'met — it was not. Treat its result as exploratory.',
    },
}

PATHOLOGY_WSI_CUT = {
    'wsi_cobra': 'Weights request-and-approval gated on HuggingFace.',
    'wsi_feather': 'All three repos need an approved access request tied to a verified institution.',
    'wsi_hibou_l': 'histai/hibou-L is gated on HuggingFace.',
    'wsi_musk': 'Nature 2025, but gated contact-sharing and CC-BY-NC-ND 4.0. '
                'https://huggingface.co/xiangjx/musk',
}

# =============================================================================
# WHERE THE PUBLISHED EVIDENCE PREDICTS A ZERO DELTA
# =============================================================================
# c13 subtracts hybrid-without from hybrid-with, against a Gaussian block of equal
# width. These are the places the literature says that subtraction lands at or below
# zero. They are calibration for reading the result, not a reason to skip a run.

EXPECTED_ZERO = {
    'nucleotide': 'The strongest signal on this list. Koo lab, Genome Biology 26 (2025), probing four '
                  'gLMs over six tasks: "probing the representations of current pre-trained gLMs do '
                  'not offer substantial advantages over conventional machine learning approaches '
                  'that use one-hot encoded sequences". Sharpest pair: MPRA zero-shot in K562, '
                  'Nucleotide Transformer Pearson 0.007 against a one-hot CNN at 0.365. '
                  'https://pmc.ncbi.nlm.nih.gov/articles/PMC12261763/ '
                  'DART-Eval, NeurIPS 2024 D&B: chromatin activity fine-tuned NT Spearman 0.515 '
                  'against ChromBPNet 0.540; variant effect AUROC 0.623 against 0.772. Verbatim: '
                  '"Simpler ab initio supervised models match or exceed the performance of much '
                  'larger, fine-tuned DNALMs". https://pmc.ncbi.nlm.nih.gov/articles/PMC12340898/ '
                  'All six models DART-Eval graded — caduceus, dnabert2, gena_lm, hyenadna, '
                  'mistral_dna, nt_v2_250m/500m — are already reps, so it grades the existing pool. '
                  'The pools also already hold kmer3-kmer6, gc_content, dinuc_freq and tetranuc_freq, '
                  'which are the baselines both papers say win.',
    'molecular': 'Praski, 25 models x 25 datasets: ChemFM mean rank 12.60 / 78.25% AUROC, COATI '
                 '13.92 / 78.02%, Uni-Mol2 13.32 / 78.11%, Uni-Mol 16.64 / 76.85% — all against ECFP '
                 'at 7.52 / 79.89%. ecfp4 is in every molecular pool, so those four predict zero or '
                 'negative. The one exception, CLAMP at 5.40 / 82.55%, has a molecule side that is an '
                 'MLP over concatenated ECFP and RDKit fingerprints, both already in the pool as '
                 'ecfp4 and rdkfp. https://arxiv.org/html/2508.06199v2',
    'protein': 'PFMBench: no sequence-only model exceeds a 50% winning rate against ESM-2 650M. '
               'VenusPLM, SaProt and xTrimoPGLM sit at exactly 50% (4 of 8 tasks), ESM-C at 38%, '
               'ESM3 1.4B at 13%. esm2 is already a rep, so for those the null is inside the pool by '
               'construction. https://arxiv.org/html/2506.14796v1',
    'antibody': 'FLAb2: protein AI models yield no statistically significant Spearman correlation on '
                '80% of developability datasets, per-property hit rates from 51% (immunogenicity) '
                'down to 14% (pharmacokinetics). At roughly 10^3 labelled points a fine-tuned one-hot '
                'encoding matches fine-tuned billion-parameter models on five of seven properties. '
                'onehot is already in ANTIBODY_REPS. '
                'https://pmc.ncbi.nlm.nih.gov/articles/PMC12767642/',
    'omics': 'scFoundation embeddings on bulk TCGA across 25 cancer types: survival C-index 0.624 '
             'against raw expression plus clinical features at 0.681; Ridge-Cox on the embeddings '
             '0.680, Lasso-Cox 0.670. gene_pca100 and gene_hvg2000 are that raw-expression baseline '
             'and are already in the pool, and scfoundation is already a rep — so this is a direct '
             'prediction of its own c13 delta.',
    'pathology_wsi': 'TCGA subtyping is saturated. NSCLC AUC spans 93.6 (CHIEF) to 99.2 (PRISM), with '
                     'mean-pooled Virchow2 at 95.8, and PRISM2\'s own text says all slide encoders '
                     'perform similarly on TCGA BRCA/NSCLC/RCC. A 31-model, 41-task benchmark found '
                     'no significant difference between pathology-specific and general vision models. '
                     'There is no headroom on tcga_luad_vs_lusc for a positive delta to show in, and '
                     'tcga_brca_pam50 has no published slide-embedding number to calibrate against '
                     'at all.',
    'reaction': 'THE ONE PLACE TO EXPECT A NON-ZERO RESULT, and it points the other way. ChemTorch on '
                'RDB7 barrier heights: DRFP with an MLP at 14.50 kcal/mol mean absolute error, '
                'condensed reaction graph with a D-MPNN at 4.10, 3D geometry with DimeNet++ at 2.54. '
                'A 3.5x error gap in favour of a learned reaction encoder over a hand-built '
                'fingerprint, opposite in sign to every other domain here.',
}

# =============================================================================
# OPEN — recorded so the absence is not silent
# =============================================================================

OPEN = {
    'lucaone': 'The only candidate that would enter the nucleotide, protein AND antibody lists as '
               'literally the same weights, which is the cleanest c13 arm available: one model, three '
               'pools, three subtractions. Nature Machine Intelligence 2025, trained across 169,861 '
               'species on DNA, RNA and protein. Code Apache-2.0 at '
               'https://github.com/LucaOne/LucaOne, 13-model HuggingFace org at '
               'https://huggingface.co/LucaGroup. NOT on the list because no page was opened giving '
               'it a metric with a named comparator and that comparator\'s number — it fails '
               'admission rule 4 as written, not on substance.',
    'rna_language_models': 'Zero RNA candidates in any lane, despite rnafm and rinalmo already being '
                           'implemented. AIDO.RNA-1.6B is public and ungated at '
                           'https://huggingface.co/genbio-ai/AIDO.RNA-1.6B (hidden size 2048, 42M '
                           'non-coding RNA sequences) but its card names no comparator with a number. '
                           'The head-to-head that exists is a comparative review of 13 RNA LMs plus '
                           '3 DNA and 1 protein control, arXiv 2505.09087, whose finding is a '
                           'trade-off rather than a winner: models strong at zero-shot secondary '
                           'structure are weaker at function classification and the reverse.',
    'unread_benchmarks': 'NABench (https://arxiv.org/abs/2511.02888), 29 nucleotide foundation models '
                         'over 162 assays and 2.6M mutated sequences — cited by nobody in this '
                         'compilation. And the Nature Methods 2026 biomedical foundation model '
                         'benchmark (s41592-026-03182-y) returns an auth redirect, so nobody has '
                         'seen it.',
}

FORWARD_COMPAT_REPS = {
    'molecular': MOLECULAR,
    'antibody': ANTIBODY,
    'protein': PROTEIN,
    'nucleotide': NUCLEOTIDE,
    'reaction': REACTION,
    'omics_transcriptomics': OMICS,
    'pathology_wsi': PATHOLOGY_WSI,
}

CUT = {
    'molecular': MOLECULAR_CUT,
    'antibody': ANTIBODY_CUT,
    'protein': PROTEIN_CUT,
    'nucleotide': NUCLEOTIDE_CUT,
    'reaction': REACTION_CUT,
    'omics_transcriptomics': OMICS_CUT,
    'pathology_wsi': PATHOLOGY_WSI_CUT,
}


# =============================================================================
# Resolving the list to rep names a loader will accept
# =============================================================================
# load_data reports one of six domains: molecular, antibody, nucleotide,
# reaction, tcga, dta. Two of those draw on more than one block of the list.
# tcga takes both the transcriptomics/methylation block and the pathology block,
# because a TCGA pool holds both. dta takes the protein block for the target
# side and the molecular block for the drug side, each under the prefix that
# domain's loader uses.

_DOMAIN_BLOCKS = {
    'molecular':  [('molecular', '')],
    'antibody':   [('antibody', '')],
    'nucleotide': [('nucleotide', '')],
    'reaction':   [('reaction', '')],
    'tcga':       [('omics_transcriptomics', ''), ('pathology_wsi', '')],
    'dta':        [('protein', 'protein_'), ('molecular', 'drug_')],
}

# Entries whose name in the list is not the name a loader would use.
# `diff_chemeleon` also has a `concat_chemeleon` form built by the same wrapper.
_EXTRA_NAMES = {
    'reaction': ['concat_chemeleon'],
}


def forward_compat_rep_names(domain):
    """Rep names to try loading as forward-compatibility candidates for `domain`.

    Returns [] for a domain with no block, so a caller can skip cleanly rather
    than raise. Names are returned in list order; a loader that has no builder
    for one of them warns and skips, and c13 reports which ones did not arrive.
    """
    blocks = _DOMAIN_BLOCKS.get(domain)
    if not blocks:
        return []
    names = []
    for block_key, prefix in blocks:
        for name in FORWARD_COMPAT_REPS[block_key]:
            names.append(prefix + name)
        for name in _EXTRA_NAMES.get(block_key, []):
            names.append(prefix + name)
    return names


def forward_compat_entry(domain, rep_name):
    """The list entry behind a loaded rep name, or None. Strips the dta prefix."""
    for block_key, prefix in _DOMAIN_BLOCKS.get(domain, []):
        bare = rep_name[len(prefix):] if prefix and rep_name.startswith(prefix) else rep_name
        entry = FORWARD_COMPAT_REPS[block_key].get(bare)
        if entry is not None:
            return entry
    return None
