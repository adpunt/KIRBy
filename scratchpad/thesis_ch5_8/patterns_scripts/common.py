SM = ['qm9','herg_ki','caco2','logd','adme_hlm','adme_rlm','adme_mdr1','herg','herg_ic50']
AB = ['flab_expression_koenig','flab_binding_koenig','flab_thermo_tresanco','abagym_EGFR','abagym_VEGF','abagym_Ang2',
      'humanness_therapeutic','ddg_combined','cdr_flexibility','abcov_kd','ginkgo_colloidal','ginkgo_hic','ginkgo_acsins','ginkgo_titer','ginkgo_pr_cho']
OTHER = {'buchwald_hartwig':'reaction','tcga_brca_pam50':'gene_expression'}
CUT = {'tcga_luad_vs_lusc','nt_H3K27me3','nt_H3K9ac','nt_H3K9me3','nt_H3K4me3','nt_promoter_tata','gb_human_enhancers_cohn','bindingdb_kd','kiba','flab_expression_adams','flab_binding_adams','suzuki','lhasa_herg','bh_doyle_plates','tz_her2_binary','tz_her2_multiclass'}
def domain(ds):
    if ds in SM: return 'small_molecule'
    if ds in AB: return 'antibody'
    return OTHER.get(ds,'?')
