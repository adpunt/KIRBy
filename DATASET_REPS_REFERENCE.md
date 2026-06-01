# DATASET_REPS reference

Per-dataset record of what's being tested: which representations are in the general / top / augmentation lists, and which precompute conditions are expected.

## How conditions are emitted

Precompute writes up to 6 condition files per dataset, with each file named `{dataset}__{tier}__{strategy}__full.pkl`:

| Condition | Emitted when |
|---|---|
| `general/none` | general rep list has ≥2 reps |
| `general/uncapped` | general ≥2 AND augmentation list non-empty |
| `top/none` | top_tier ≥2 reps |
| `top/uncapped` | top ≥2 AND augmentation non-empty |
| `top/capped` | top ≥2 AND augmentation non-empty, EXCEPT for the skip list below |
| `top+borderline/uncapped` | top ≥2 AND augmentation non-empty AND general has reps outside top |

Edge cases: if general has exactly 1 rep + augmentation, only `general/uncapped` is emitted. Same for top.

## top/capped skip list

`top/capped` is **not** emitted for these datasets — the capped strategy inflates training set so far that the condition projects past the 7-day partition limit (see commit `706dea2`):

- `ddg_combined`
- `nt_H3K4me3`
- `tcga_brca_pam50`
- `tcga_luad_vs_lusc`

## MOLECULAR

### `adme_hlm`

- **General** (16 reps): `mordred`, `pdv`, `mhg_gnn`, `smited`, `molformer`, `grover`, `mol2vec`, `chemberta`, `atom_pair`, `chembert`, `graph_kernel`, `avalon`, `top_torsion`, `ecfp4`, `selformer`, `graphmvp`
- **Top tier** (6 reps): `mordred`, `pdv`, `mhg_gnn`, `smited`, `molformer`, `grover`
- **Augmentation** (5 reps): `pharmacophore3d`, `maccs`, `usrcat`, `coulomb_matrix`, `usr`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `adme_mdr1`

- **General** (16 reps): `mordred`, `pdv`, `mhg_gnn`, `avalon`, `grover`, `smited`, `molformer`, `atom_pair`, `graph_kernel`, `maccs`, `chemberta`, `ecfp4`, `chembert`, `selformer`, `top_torsion`, `rdkfp`
- **Top tier** (5 reps): `mordred`, `pdv`, `mhg_gnn`, `avalon`, `grover`
- **Augmentation** (2 reps): `pharmacophore3d`, `coulomb_matrix`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `adme_rlm`

- **General** (16 reps): `pdv`, `mhg_gnn`, `smited`, `mordred`, `molformer`, `grover`, `mol2vec`, `chemberta`, `chembert`, `graph_kernel`, `atom_pair`, `selformer`, `avalon`, `ecfp4`, `top_torsion`, `graphmvp`
- **Top tier** (6 reps): `pdv`, `mhg_gnn`, `smited`, `mordred`, `molformer`, `grover`
- **Augmentation** (5 reps): `pharmacophore3d`, `maccs`, `usrcat`, `coulomb_matrix`, `usr`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `caco2`

- **General** (12 reps): `graph_kernel`, `mol2vec`, `mhg_gnn`, `ecfp4`, `pdv`, `grover`, `avalon`, `mordred`, `top_torsion`, `smited`, `atom_pair`, `molformer`
- **Top tier** (8 reps): `graph_kernel`, `mol2vec`, `mhg_gnn`, `ecfp4`, `pdv`, `grover`, `avalon`, `mordred`
- **Augmentation** (5 reps): `maccs`, `pharmacophore3d`, `usrcat`, `coulomb_matrix`, `usr`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `herg`

- **General** (18 reps): `mhg_gnn`, `mordred`, `avalon`, `ecfp4`, `grover`, `atom_pair`, `pdv`, `graph_kernel`, `mol2vec`, `top_torsion`, `maccs`, `smited`, `molformer`, `chembert`, `rdkfp`, `selformer`, `chemberta`, `graphmvp`
- **Top tier** (8 reps): `mhg_gnn`, `mordred`, `avalon`, `ecfp4`, `grover`, `atom_pair`, `pdv`, `graph_kernel`
- **Augmentation** (3 reps): `usrcat`, `coulomb_matrix`, `molclr`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `herg_ki`

- **General** (15 reps): `mhg_gnn`, `mordred`, `graph_kernel`, `avalon`, `grover`, `pdv`, `mol2vec`, `ecfp4`, `molformer`, `chemberta`, `atom_pair`, `chembert`, `smited`, `top_torsion`, `graphmvp`
- **Top tier** (8 reps): `mhg_gnn`, `mordred`, `graph_kernel`, `avalon`, `grover`, `pdv`, `mol2vec`, `ecfp4`
- **Augmentation** (5 reps): `maccs`, `pharmacophore3d`, `coulomb_matrix`, `usr`, `usrcat`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `logd`

- **General** (15 reps): `mhg_gnn`, `mordred`, `grover`, `mol2vec`, `pdv`, `ecfp4`, `smited`, `molformer`, `avalon`, `atom_pair`, `chemberta`, `top_torsion`, `chembert`, `graph_kernel`, `graphmvp`
- **Top tier** (8 reps): `mhg_gnn`, `mordred`, `grover`, `mol2vec`, `pdv`, `ecfp4`, `smited`, `molformer`
- **Augmentation** (6 reps): `selformer`, `rdkfp`, `maccs`, `pharmacophore3d`, `molclr`, `usrcat`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `qm9`

- **General** (16 reps): `mordred`, `pdv`, `grover`, `mhg_gnn`, `avalon`, `rdkfp`, `atom_pair`, `smited`, `ecfp4`, `selformer`, `maccs`, `chemberta`, `mol2vec`, `chembert`, `graphmvp`, `molformer`
- **Top tier** (5 reps): `mordred`, `pdv`, `grover`, `mhg_gnn`, `avalon`
- **Augmentation** (4 reps): `top_torsion`, `pharmacophore3d`, `usrcat`, `coulomb_matrix`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

## ANTIBODY

### `abagym_Ang2`

- **General** (24 reps): `aac`, `abroberta`, `abroberta_cdr`, `antiberty`, `antiberty_cdr`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `mutation_features`, `structural`
- **Top tier** (4 reps): `saprot`, `esm3`, `esm2_midlayers`, `esm2_layer24`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `abagym_EGFR`

- **General** (26 reps): `aac`, `ablang2`, `abroberta`, `abroberta_cdr`, `antiberty`, `antiberty_cdr`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `mutation_features`, `structural`
- **Top tier** (5 reps): `antiberty`, `antiberty_cdr`, `esm2_layer24`, `esm2_early`, `esm2_midlayers`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `abagym_HER2`

- **General** (13 reps): `prot_t5_cdr`, `prot_t5`, `esm2_early`, `esm2`, `esmc`, `ablang2`, `igt5`, `antiberty`, `esm2_midlayers`, `esm2_layer24`, `saprot`, `aac`, `mutation_features`
- **Top tier** (3 reps): `mutation_features`, `prot_t5_cdr`, `prot_t5`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `abagym_VEGF`

- **General** (24 reps): `aac`, `ablang2`, `abroberta`, `abroberta_cdr`, `antiberty`, `antiberty_cdr`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `mutation_features`, `structural`
- **Top tier** (4 reps): `mutation_features`, `aac`, `prot_t5_cdr`, `prot_t5`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `abagym_lysozyme`

- **General** (3 reps): `mutation_features`, `aac`, `blosum`
- **Top tier** (1 reps): `mutation_features`
- **Augmentation** (2 reps): `structural`, `antiberty_llr`
- **Expected conditions** (3): `general/none`, `general/uncapped`, `top/uncapped`

### `abcov_kd`

- **General** (27 reps): `igt5`, `esm2_per_cdr`, `prot_t5`, `esm2_per_region`, `antiberty`, `esm2_midlayers`, `esm2_cdr3`, `esm2_cdr`, `abroberta`, `esmc_cdr`, `esm2`, `blosum`, `ablang2`, `onehot`, `abroberta_cdr`, `saprot`, `esm2_layer24`, `prot_t5_cdr`, `saprot_cdr`, `esm3`, `esmc`, `antiberty_cdr`, `aac`, `esm3_cdr`, `esm2_early`, `esm2_concat`, `esm2_entropy_top20`
- **Top tier** (8 reps): `esm2_entropy_top20`, `prot_t5`, `blosum`, `antiberty`, `esm2_midlayers`, `abroberta`, `esm2`, `ablang2`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `ddg_combined`

- **General** (18 reps): `esm2_wt_delta`, `prot_t5`, `esm2_midlayers`, `esm2_concat`, `onehot`, `prot_t5_cdr`, `antiberty_cdr`, `esmc`, `mutation_features`, `saprot`, `esmc_cdr`, `antiberty`, `esm2_cdr`, `esm2_cdr3`, `esm2`, `esm2_layer24`, `saprot_cdr`, `esm2_per_cdr`
- **Top tier** (3 reps): `esm2_wt_delta`, `prot_t5`, `esm2_midlayers`
- **Augmentation** (1 reps): `aac`
- **Expected conditions** (5): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top+borderline/uncapped`
  - _`top/capped` skipped — see header._

### `flab_aggregation_jain`

- **General** (23 reps): `esm2_concat`, `ablang2`, `igt5`, `prot_t5_cdr`, `prot_t5`, `saprot`, `saprot_cdr`, `esm2_early`, `mutation_features`, `esmc`, `esmc_cdr`, `antiberty`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `antiberty_cdr`, `esm2_entropy_top20`, `abroberta`, `abroberta_cdr`, `esm2_mutsite`, `esm3`, `esm3_cdr`, `esm2_wt_delta`
- **Top tier** (3 reps): `esm2_concat`, `ablang2`, `igt5`
- **Augmentation** (2 reps): `paratope`, `structural`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_aggregation_jain_hicrt`

- **General** (23 reps): `prot_t5`, `prot_t5_cdr`, `esm2_early`, `esm2_mutsite`, `esm2_concat`, `esmc`, `esmc_cdr`, `esm3`, `esm3_cdr`, `esm2_layer24`, `ablang2`, `saprot`, `saprot_cdr`, `esm2_per_region`, `esm2_midlayers`, `blosum`, `esm2_wt_delta`, `esm2_cdr`, `esm2_cdr3`, `aac`, `esm2`, `esm2_per_cdr`, `onehot`
- **Top tier** (3 reps): `prot_t5`, `prot_t5_cdr`, `esm2_early`
- **Augmentation** (3 reps): `paratope`, `structural`, `mutation_features`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_binding_adams`

- **General** (26 reps): `esm2`, `esmc`, `ablang2`, `prot_t5`, `igt5`, `onehot`, `blosum`, `esm2_cdr`, `prot_t5_cdr`, `esm2_midlayers`, `esm2_layer24`, `esm2_early`, `esm2_concat`, `esm2_entropy_top20`, `abroberta`, `abroberta_cdr`, `esm3`, `saprot`, `esm2_cdr3`, `esm2_per_cdr`, `esm2_per_region`, `esm2_wt_delta`, `esm2_mutsite`, `saprot_cdr`, `esmc_cdr`, `esm3_cdr`
- **Top tier** (8 reps): `blosum`, `onehot`, `esm3`, `esm3_cdr`, `saprot`, `saprot_cdr`, `esm2_cdr`, `esm2_cdr3`
- **Augmentation** (2 reps): `aac`, `paratope`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_binding_koenig`

- **General** (24 reps): `esm2`, `esmc`, `ablang2`, `antiberty`, `prot_t5`, `igt5`, `esm2_cdr`, `prot_t5_cdr`, `esm2_midlayers`, `esm2_layer24`, `esm2_early`, `esm2_concat`, `esm2_entropy_top20`, `abroberta`, `abroberta_cdr`, `esm3`, `saprot`, `esm2_cdr3`, `esm2_per_cdr`, `esm2_per_region`, `esm2_wt_delta`, `saprot_cdr`, `esmc_cdr`, `esm3_cdr`
- **Top tier** (8 reps): `ablang2`, `saprot`, `saprot_cdr`, `esm3`, `esm3_cdr`, `prot_t5`, `prot_t5_cdr`, `igt5`
- **Augmentation** (4 reps): `esm2_llr`, `mutation_features`, `antiberty_llr`, `aac`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_binding_phillips_h3`

- **General** (28 reps): `esm2`, `esmc`, `ablang2`, `antiberty`, `prot_t5`, `igt5`, `onehot`, `blosum`, `esm2_cdr`, `antiberty_cdr`, `prot_t5_cdr`, `esm2_midlayers`, `esm2_layer24`, `esm2_early`, `esm2_concat`, `esm2_entropy_top20`, `abroberta`, `abroberta_cdr`, `esm3`, `saprot`, `esm2_cdr3`, `esm2_per_cdr`, `esm2_per_region`, `esm2_wt_delta`, `esm2_mutsite`, `saprot_cdr`, `esmc_cdr`, `esm3_cdr`
- **Top tier** (4 reps): `onehot`, `blosum`, `igt5`, `esm2_early`
- **Augmentation** (3 reps): `mutation_features`, `paratope`, `aac`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_expression_adams`

- **General** (28 reps): `esm2`, `esmc`, `ablang2`, `antiberty`, `prot_t5`, `igt5`, `onehot`, `blosum`, `esm2_cdr`, `antiberty_cdr`, `prot_t5_cdr`, `esm2_midlayers`, `esm2_layer24`, `esm2_early`, `esm2_concat`, `esm2_entropy_top20`, `abroberta`, `abroberta_cdr`, `esm3`, `saprot`, `esm2_cdr3`, `esm2_per_cdr`, `esm2_per_region`, `esm2_wt_delta`, `esm2_mutsite`, `saprot_cdr`, `esmc_cdr`, `esm3_cdr`
- **Top tier** (7 reps): `blosum`, `saprot`, `saprot_cdr`, `esm3`, `esm3_cdr`, `prot_t5`, `prot_t5_cdr`
- **Augmentation** (1 reps): `esm2_llr`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_expression_koenig`

- **General** (26 reps): `esm2`, `esmc`, `ablang2`, `antiberty`, `prot_t5`, `igt5`, `esm2_cdr`, `antiberty_cdr`, `prot_t5_cdr`, `esm2_midlayers`, `esm2_layer24`, `esm2_early`, `esm2_concat`, `esm2_entropy_top20`, `abroberta`, `abroberta_cdr`, `esm3`, `saprot`, `esm2_cdr3`, `esm2_per_cdr`, `esm2_per_region`, `esm2_wt_delta`, `esm2_mutsite`, `saprot_cdr`, `esmc_cdr`, `esm3_cdr`
- **Top tier** (7 reps): `esm3`, `esm3_cdr`, `prot_t5_cdr`, `prot_t5`, `saprot`, `saprot_cdr`, `igt5`
- **Augmentation** (3 reps): `mutation_features`, `esm2_llr`, `aac`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_immunogenicity_marks`

- **General** (27 reps): `antiberty_cdr`, `antiberty`, `igt5`, `prot_t5`, `prot_t5_cdr`, `abroberta_cdr`, `esm2_early`, `abroberta`, `ablang2`, `onehot`, `saprot_cdr`, `saprot`, `esm2_midlayers`, `blosum`, `esm2`, `esm2_cdr3`, `esm2_cdr`, `esm2_layer24`, `esmc_cdr`, `esmc`, `aac`, `esm2_concat`, `esm3_cdr`, `esm3`, `esm2_per_cdr`, `esm2_per_region`, `esm2_entropy_top20`
- **Top tier** (8 reps): `antiberty_cdr`, `antiberty`, `esm2_entropy_top20`, `igt5`, `prot_t5`, `prot_t5_cdr`, `abroberta_cdr`, `abroberta`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `flab_pk_jain_fcrnrt`

- **General** (11 reps): `ablang2`, `igt5`, `esm2_early`, `prot_t5`, `saprot`, `saprot_cdr`, `mutation_features`, `prot_t5_cdr`, `esm2_concat`, `esm2_mutsite`, `antiberty_cdr`
- **Top tier** (2 reps): `ablang2`, `igt5`
- **Augmentation** (1 reps): `aac`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_pk_jain_heprt`

- **General** (20 reps): `igt5`, `mutation_features`, `esm2_concat`, `ablang2`, `prot_t5`, `prot_t5_cdr`, `aac`, `esm2_cdr`, `esm2_cdr3`, `esm2_per_cdr`, `esm2_per_region`, `esm2_mutsite`, `esm2`, `saprot`, `saprot_cdr`, `esmc`, `esmc_cdr`, `esm2_wt_delta`, `esm2_early`, `esm2_midlayers`
- **Top tier** (6 reps): `igt5`, `mutation_features`, `esm2_concat`, `ablang2`, `prot_t5`, `prot_t5_cdr`
- **Augmentation** (2 reps): `paratope`, `structural`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_polyreact_jain_bvp`

- **General** (23 reps): `ablang2`, `igt5`, `antiberty`, `antiberty_cdr`, `esm2_concat`, `esm2_mutsite`, `abroberta`, `abroberta_cdr`, `esm2_per_region`, `prot_t5_cdr`, `prot_t5`, `esmc`, `esmc_cdr`, `esm2_cdr`, `esm2_cdr3`, `esm2`, `blosum`, `esm2_per_cdr`, `onehot`, `paratope`, `structural`, `esm2_early`, `esm2_wt_delta`
- **Top tier** (4 reps): `ablang2`, `igt5`, `antiberty`, `antiberty_cdr`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `flab_polyreact_jain_cic`

- **General** (21 reps): `prot_t5`, `prot_t5_cdr`, `esm2_early`, `igt5`, `esm2_concat`, `ablang2`, `saprot`, `saprot_cdr`, `esm2_midlayers`, `aac`, `esm2_layer24`, `mutation_features`, `esm2_per_region`, `esm2_per_cdr`, `esm2_cdr`, `esm2_cdr3`, `esm2`, `esm2_wt_delta`, `esmc`, `esmc_cdr`, `esm2_mutsite`
- **Top tier** (3 reps): `prot_t5`, `prot_t5_cdr`, `esm2_early`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `flab_thermo_jain`

- **General** (22 reps): `ablang2`, `abroberta`, `abroberta_cdr`, `igt5`, `esm3`, `esm3_cdr`, `esm2_cdr`, `esm2_cdr3`, `esm2_per_region`, `esm2_wt_delta`, `esm2_per_cdr`, `antiberty`, `esm2`, `saprot`, `saprot_cdr`, `esmc`, `esmc_cdr`, `esm2_entropy_top20`, `esm2_layer24`, `antiberty_cdr`, `prot_t5`, `prot_t5_cdr`
- **Top tier** (4 reps): `ablang2`, `abroberta`, `igt5`, `esm3`
- **Augmentation** (1 reps): `aac`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `flab_thermo_tresanco`

- **General** (28 reps): `esm3`, `esm3_cdr`, `saprot_cdr`, `saprot`, `blosum`, `esmc`, `esmc_cdr`, `prot_t5`, `igt5`, `onehot`, `prot_t5_cdr`, `esm2_early`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_wt_delta`, `esm2_per_cdr`, `esm2_entropy_top20`, `esm2`, `esm2_per_region`, `esm2_midlayers`, `antiberty`, `antiberty_cdr`, `aac`, `esm2_layer24`, `esm2_mutsite`, `paratope`, `structural`
- **Top tier** (9 reps): `esm3`, `saprot`, `blosum`, `esmc`, `prot_t5`, `igt5`, `onehot`, `prot_t5_cdr`, `esm2_early`
- **Augmentation** (1 reps): `mutation_features`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `ginkgo_acsins`

- **General** (26 reps): `aac`, `ablang2`, `abroberta`, `abroberta_cdr`, `antiberty`, `antiberty_cdr`, `blosum`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `esm2_entropy_top20`
- **Top tier** (2 reps): `igt5`, `ablang2`
- **Augmentation** (2 reps): `paratope`, `structural`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `ginkgo_colloidal`

- **General** (23 reps): `ablang2`, `antiberty`, `antiberty_cdr`, `blosum`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `esm2_entropy_top20`
- **Top tier** (2 reps): `prot_t5`, `prot_t5_cdr`
- **Augmentation** (2 reps): `paratope`, `structural`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `ginkgo_heparin`

- **General** (27 reps): `aac`, `ablang2`, `abroberta`, `abroberta_cdr`, `antiberty`, `antiberty_cdr`, `blosum`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `onehot`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `esm2_entropy_top20`
- **Top tier** (4 reps): `esm2_concat`, `prot_t5`, `prot_t5_cdr`, `ablang2`
- **Augmentation** (2 reps): `paratope`, `structural`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `ginkgo_hic`

- **General** (21 reps): `aac`, `ablang2`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `esm2_entropy_top20`
- **Top tier** (3 reps): `esm2_concat`, `prot_t5`, `prot_t5_cdr`
- **Augmentation** (2 reps): `paratope`, `structural`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `ginkgo_pr_cho`

- **General** (23 reps): `aac`, `ablang2`, `blosum`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `onehot`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `esm2_entropy_top20`
- **Top tier** (2 reps): `prot_t5`, `prot_t5_cdr`
- **Augmentation** (2 reps): `paratope`, `structural`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `ginkgo_titer`

- **General** (23 reps): `ablang2`, `abroberta`, `abroberta_cdr`, `antiberty`, `antiberty_cdr`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_layer24`, `esm2_midlayers`, `esm2_per_cdr`, `esm2_per_region`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`, `esm2_entropy_top20`
- **Top tier** (2 reps): `saprot`, `saprot_cdr`
- **Augmentation** (2 reps): `paratope`, `structural`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `humanness_therapeutic`

- **General** (26 reps): `antiberty_cdr`, `antiberty`, `igt5`, `abroberta_cdr`, `abroberta`, `ablang2`, `blosum`, `onehot`, `esm2_per_cdr`, `esm2_cdr3`, `esm2_cdr`, `esm2_concat`, `esm2`, `esm2_per_region`, `esm2_midlayers`, `esmc_cdr`, `esmc`, `esm2_entropy_top20`, `prot_t5`, `prot_t5_cdr`, `esm2_layer24`, `esm3_cdr`, `esm3`, `saprot_cdr`, `saprot`, `esm2_early`
- **Top tier** (21 reps): `antiberty_cdr`, `antiberty`, `igt5`, `abroberta_cdr`, `abroberta`, `ablang2`, `blosum`, `onehot`, `esm2_per_cdr`, `esm2_cdr3`, `esm2_cdr`, `esm2_concat`, `esm2`, `esm2_per_region`, `esm2_midlayers`, `esmc_cdr`, `esmc`, `esm2_entropy_top20`, `prot_t5`, `prot_t5_cdr`, `esm2_layer24`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `tz_her2_binary`

- **General** (29 reps): `aac`, `ablang2`, `abroberta`, `abroberta_cdr`, `antiberty`, `blosum`, `esm2`, `esm2_cdr`, `esm2_cdr3`, `esm2_concat`, `esm2_early`, `esm2_layer24`, `esm2_midlayers`, `esm2_mutsite`, `esm2_per_cdr`, `esm2_per_region`, `esm2_wt_delta`, `esm2_entropy_top20`, `esm3`, `esm3_cdr`, `esmc`, `esmc_cdr`, `igt5`, `mutation_features`, `onehot`, `prot_t5`, `prot_t5_cdr`, `saprot`, `saprot_cdr`
- **Top tier** (2 reps): `blosum`, `onehot`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

### `tz_her2_multiclass`

- **General** (16 reps): `blosum`, `onehot`, `esm2_concat`, `esm2_per_region`, `esm2_per_cdr`, `esm2_wt_delta`, `esm2`, `esm2_cdr3`, `esm2_cdr`, `esm2_midlayers`, `igt5`, `esm2_early`, `esm2_layer24`, `abroberta`, `abroberta_cdr`, `esm2_entropy_top20`
- **Top tier** (3 reps): `blosum`, `onehot`, `esm2_concat`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

## DTA

### `bindingdb_kd`

- **General** (16 reps): `drug_pdv`, `drug_mordred`, `drug_maccs`, `drug_chemberta`, `drug_avalon`, `drug_molformer`, `drug_selformer`, `drug_ecfp4`, `drug_mol2vec`, `protein_esm2_midlayers`, `protein_esm3`, `protein_esm2`, `protein_esm2_pocket`, `protein_saprot`, `protein_esmc`, `protein_kmer`
- **Top tier** (4 reps): `drug_pdv`, `drug_mordred`, `protein_esm2_midlayers`, `protein_esm3`
- **Augmentation** (3 reps): `protein_aac`, `protein_pocket`, `protein_contact_graph`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `kiba`

- **General** (16 reps): `drug_mordred`, `drug_pdv`, `drug_mol2vec`, `drug_chemberta`, `drug_ecfp4`, `drug_molformer`, `drug_avalon`, `drug_maccs`, `drug_selformer`, `protein_saprot`, `protein_esm2_pocket`, `protein_esmc`, `protein_esm2_midlayers`, `protein_esm3`, `protein_esm2`, `protein_kmer`
- **Top tier** (3 reps): `drug_mordred`, `drug_pdv`, `protein_saprot`
- **Augmentation** (3 reps): `protein_usr_shape`, `protein_pocket`, `protein_contact_graph`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

## REACTION

### `bh_doyle_plates`

- **General** (18 reps): `reactant_avalon`, `diff_avalon`, `diff_mhg_gnn`, `diff_ecfp4`, `reactant_grover`, `reactant_ecfp4`, `cgr_morgan`, `reactant_mhg_gnn`, `diff_grover`, `drfp_2048`, `reactiont5`, `diff_mol2vec`, `onehot_components`, `reactant_mol2vec`, `rxnfp`, `diff_maccs`, `reactant_mordred`, `diff_mordred`
- **Top tier** (7 reps): `reactant_avalon`, `diff_avalon`, `diff_mhg_gnn`, `diff_ecfp4`, `reactant_grover`, `reactant_ecfp4`, `cgr_morgan`
- **Augmentation** (4 reps): `product_maccs`, `reactant_maccs`, `product_pharmacophore3d`, `reactant_pharmacophore3d`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `buchwald_hartwig`

- **General** (8 reps): `reactant_ecfp4`, `diff_ecfp4`, `drfp_2048`, `reactiont5`, `onehot_components`, `reactant_pdv`, `diff_maccs`, `rxnfp`
- **Top tier** (3 reps): `reactant_ecfp4`, `diff_ecfp4`, `drfp_2048`
- **Augmentation** (0 reps): _(empty — no aug conditions emitted)_
- **Expected conditions** (2): `general/none`, `top/none`

## TCGA

### `tcga_brca_pam50`

- **General** (21 reps): `gene_geneformer_v2`, `gene_geneformer`, `gene_bulkrnabert`, `rppa`, `cnv_pca50`, `wsi_titan`, `methyl_pca50`, `wsi_hoptimus0`, `wsi_gigapath`, `mirna_top200`, `methyl_hvg`, `wsi_moco`, `wsi_phikon`, `cnv_gistic`, `wsi_ctranspath`, `mutation_matrix`, `mut_pathway`, `mut_driver`, `mut_pca50`, `stemness_mrnasi`, `immune_cibersort`
- **Top tier** (3 reps): `gene_geneformer_v2`, `gene_geneformer`, `gene_bulkrnabert`
- **Augmentation** (6 reps): `hrd_score`, `mut_tmb`, `purity_consensus`, `gene_stats`, `gene_chromosome_arm`, `immune_xcell`
- **Expected conditions** (5): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top+borderline/uncapped`
  - _`top/capped` skipped — see header._

### `tcga_luad_vs_lusc`

- **General** (20 reps): `gene_geneformer_v2`, `gene_bulkrnabert`, `methyl_hvg`, `gene_geneformer`, `gene_chromosome_arm`, `wsi_titan`, `methyl_pca50`, `cnv_pca50`, `mirna_top200`, `wsi_gigapath`, `wsi_hoptimus0`, `cnv_gistic`, `wsi_ctranspath`, `wsi_moco`, `wsi_phikon`, `rppa`, `mutation_matrix`, `mut_driver`, `mut_pca50`, `mut_pathway`
- **Top tier** (8 reps): `gene_geneformer_v2`, `gene_bulkrnabert`, `methyl_hvg`, `gene_geneformer`, `gene_chromosome_arm`, `wsi_titan`, `methyl_pca50`, `cnv_pca50`
- **Augmentation** (8 reps): `immune_xcell`, `gene_stats`, `stemness_mrnasi`, `immune_cibersort`, `cimp_status`, `mut_tmb`, `hrd_score`, `purity_consensus`
- **Expected conditions** (5): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top+borderline/uncapped`
  - _`top/capped` skipped — see header._

## NUCLEOTIDE

### `gb_demo_coding_vs_intergenomic_seqs`

- **General** (11 reps): `nt_v2_250m`, `nt_v2_500m`, `kmer5`, `grover_dna`, `kmer6`, `mistral_dna`, `kmer4`, `tetranuc_freq`, `hyenadna`, `kmer3`, `tnf_canonical`
- **Top tier** (5 reps): `nt_v2_250m`, `nt_v2_500m`, `kmer5`, `grover_dna`, `kmer6`
- **Augmentation** (1 reps): `codon_usage`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `gb_demo_human_or_worm`

- **General** (11 reps): `tetranuc_freq`, `tnf_canonical`, `kmer4`, `kmer5`, `grover_dna`, `kmer3`, `nt_v2_500m`, `nt_v2_250m`, `mistral_dna`, `kmer6`, `hyenadna`
- **Top tier** (8 reps): `tetranuc_freq`, `tnf_canonical`, `kmer4`, `kmer5`, `grover_dna`, `kmer3`, `nt_v2_500m`, `nt_v2_250m`
- **Augmentation** (1 reps): `codon_usage`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `gb_human_enhancers_cohn`

- **General** (10 reps): `grover_dna`, `kmer6`, `kmer5`, `mistral_dna`, `tnf_canonical`, `hyenadna`, `nt_v2_250m`, `kmer4`, `nt_v2_500m`, `tetranuc_freq`
- **Top tier** (4 reps): `grover_dna`, `kmer6`, `kmer5`, `mistral_dna`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `gb_human_nontata_promoters`

- **General** (9 reps): `kmer5`, `kmer6`, `kmer4`, `tetranuc_freq`, `grover_dna`, `tnf_canonical`, `dnabert2`, `mistral_dna`, `kmer3`
- **Top tier** (4 reps): `kmer5`, `kmer6`, `kmer4`, `tetranuc_freq`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H2AFZ`

- **General** (11 reps): `mistral_dna`, `hyenadna`, `grover_dna`, `kmer6`, `nt_v2_250m`, `nt_v2_500m`, `tnf_canonical`, `kmer4`, `kmer5`, `tetranuc_freq`, `kmer3`
- **Top tier** (4 reps): `mistral_dna`, `hyenadna`, `grover_dna`, `kmer6`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H3K27ac`

- **General** (10 reps): `kmer6`, `grover_dna`, `mistral_dna`, `kmer5`, `tnf_canonical`, `nt_v2_500m`, `nt_v2_250m`, `kmer4`, `hyenadna`, `tetranuc_freq`
- **Top tier** (5 reps): `kmer6`, `grover_dna`, `mistral_dna`, `kmer5`, `tnf_canonical`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H3K27me3`

- **General** (11 reps): `mistral_dna`, `grover_dna`, `hyenadna`, `nt_v2_250m`, `nt_v2_500m`, `tnf_canonical`, `kmer4`, `kmer3`, `kmer5`, `tetranuc_freq`, `kmer6`
- **Top tier** (5 reps): `mistral_dna`, `grover_dna`, `hyenadna`, `nt_v2_250m`, `nt_v2_500m`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H3K36me3`

- **General** (11 reps): `mistral_dna`, `hyenadna`, `nt_v2_250m`, `nt_v2_500m`, `grover_dna`, `kmer3`, `kmer4`, `kmer5`, `kmer6`, `tetranuc_freq`, `tnf_canonical`
- **Top tier** (5 reps): `mistral_dna`, `hyenadna`, `nt_v2_250m`, `nt_v2_500m`, `grover_dna`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H3K4me1`

- **General** (11 reps): `grover_dna`, `mistral_dna`, `hyenadna`, `nt_v2_250m`, `kmer6`, `kmer5`, `nt_v2_500m`, `tnf_canonical`, `kmer4`, `tetranuc_freq`, `kmer3`
- **Top tier** (5 reps): `grover_dna`, `mistral_dna`, `hyenadna`, `nt_v2_250m`, `kmer6`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H3K4me2`

- **General** (11 reps): `mistral_dna`, `hyenadna`, `grover_dna`, `kmer6`, `nt_v2_500m`, `nt_v2_250m`, `kmer5`, `tnf_canonical`, `kmer4`, `tetranuc_freq`, `kmer3`
- **Top tier** (5 reps): `mistral_dna`, `hyenadna`, `grover_dna`, `kmer6`, `nt_v2_500m`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H3K4me3`

- **General** (12 reps): `grover_dna`, `mistral_dna`, `hyenadna`, `nt_v2_500m`, `tnf_canonical`, `kmer6`, `nt_v2_250m`, `kmer5`, `kmer3`, `kmer4`, `tetranuc_freq`, `dnabert2`
- **Top tier** (6 reps): `grover_dna`, `mistral_dna`, `hyenadna`, `nt_v2_500m`, `tnf_canonical`, `kmer6`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (5): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top+borderline/uncapped`
  - _`top/capped` skipped — see header._

### `nt_H3K9ac`

- **General** (12 reps): `kmer6`, `kmer5`, `mistral_dna`, `grover_dna`, `nt_v2_500m`, `nt_v2_250m`, `hyenadna`, `tnf_canonical`, `kmer4`, `tetranuc_freq`, `kmer3`, `dnabert2`
- **Top tier** (4 reps): `kmer6`, `kmer5`, `mistral_dna`, `grover_dna`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H3K9me3`

- **General** (11 reps): `mistral_dna`, `hyenadna`, `grover_dna`, `nt_v2_250m`, `nt_v2_500m`, `kmer4`, `kmer6`, `tnf_canonical`, `tetranuc_freq`, `kmer5`, `kmer3`
- **Top tier** (3 reps): `mistral_dna`, `hyenadna`, `grover_dna`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_H4K20me1`

- **General** (11 reps): `hyenadna`, `mistral_dna`, `grover_dna`, `nt_v2_250m`, `nt_v2_500m`, `kmer3`, `kmer4`, `tetranuc_freq`, `tnf_canonical`, `kmer6`, `kmer5`
- **Top tier** (3 reps): `hyenadna`, `mistral_dna`, `grover_dna`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_enhancers`

- **General** (12 reps): `grover_dna`, `mistral_dna`, `kmer6`, `hyenadna`, `nt_v2_500m`, `nt_v2_250m`, `kmer5`, `tetranuc_freq`, `kmer4`, `tnf_canonical`, `kmer3`, `dnabert2`
- **Top tier** (5 reps): `grover_dna`, `mistral_dna`, `kmer6`, `hyenadna`, `nt_v2_500m`
- **Augmentation** (4 reps): `dinuc_freq`, `gc_content`, `codon_usage`, `complexity`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_enhancers_types`

- **General** (12 reps): `mistral_dna`, `grover_dna`, `kmer6`, `hyenadna`, `nt_v2_500m`, `nt_v2_250m`, `kmer5`, `kmer4`, `tnf_canonical`, `tetranuc_freq`, `kmer3`, `dnabert2`
- **Top tier** (5 reps): `mistral_dna`, `grover_dna`, `kmer6`, `hyenadna`, `nt_v2_500m`
- **Augmentation** (1 reps): `codon_usage`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_promoter_all`

- **General** (12 reps): `mistral_dna`, `hyenadna`, `grover_dna`, `nt_v2_250m`, `kmer6`, `kmer5`, `nt_v2_500m`, `kmer4`, `tnf_canonical`, `tetranuc_freq`, `kmer3`, `dnabert2`
- **Top tier** (3 reps): `mistral_dna`, `hyenadna`, `grover_dna`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_promoter_no_tata`

- **General** (12 reps): `mistral_dna`, `hyenadna`, `grover_dna`, `nt_v2_500m`, `kmer6`, `nt_v2_250m`, `kmer5`, `kmer4`, `tnf_canonical`, `tetranuc_freq`, `kmer3`, `dnabert2`
- **Top tier** (4 reps): `mistral_dna`, `hyenadna`, `grover_dna`, `nt_v2_500m`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_promoter_tata`

- **General** (10 reps): `mistral_dna`, `kmer5`, `hyenadna`, `grover_dna`, `kmer4`, `nt_v2_250m`, `kmer6`, `tnf_canonical`, `tetranuc_freq`, `nt_v2_500m`
- **Top tier** (2 reps): `mistral_dna`, `kmer5`
- **Augmentation** (5 reps): `codon_usage`, `complexity`, `dinuc_freq`, `dna_shape`, `gc_content`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_splice_sites_acceptors`

- **General** (11 reps): `nt_v2_250m`, `grover_dna`, `nt_v2_500m`, `mistral_dna`, `hyenadna`, `kmer6`, `kmer4`, `kmer3`, `kmer5`, `tetranuc_freq`, `tnf_canonical`
- **Top tier** (4 reps): `nt_v2_250m`, `grover_dna`, `nt_v2_500m`, `mistral_dna`
- **Augmentation** (1 reps): `codon_usage`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

### `nt_splice_sites_donors`

- **General** (11 reps): `nt_v2_250m`, `nt_v2_500m`, `grover_dna`, `mistral_dna`, `hyenadna`, `kmer6`, `kmer5`, `kmer4`, `tetranuc_freq`, `kmer3`, `tnf_canonical`
- **Top tier** (2 reps): `nt_v2_250m`, `nt_v2_500m`
- **Augmentation** (1 reps): `codon_usage`
- **Expected conditions** (6): `general/none`, `general/uncapped`, `top/none`, `top/uncapped`, `top/capped`, `top+borderline/uncapped`

