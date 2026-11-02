# f37 herg_ki

Metric rmse (lower is better). Drop table: mean over folds and draws of damaged score minus clean score, sign flipped when higher is better, so positive = worse.

Correlations: Pearson between held-out predictions of two kinds on the same 1412 molecules, mean over the pairs. Diagonal = same kind, draw 0 vs 1.

## Level position 0

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.942, outlier_only_p10 0.955, outlier_p10 0.946, gaussian 0.953, student_t_nu3 0.959, censoring 1.000, outlier_only_p05 0.960, outlier_p05 0.951, grouped_wider 0.950, laplace 0.953, outlier_only_p01 0.962, outlier_p01 0.949, student_t_nu5 0.953, student_t_nu10 0.947, shift_only_p05 0.985, shift_only_p10 0.977, sparse_dprime_p02 0.984, sparse_dprime_p05 0.981, sparse_dprime_p10 0.972, sparse_dprime_up_p02 0.984, sparse_dprime_up_p05 0.976, sparse_dprime_up_p10 0.971

## Level position 1

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.780, outlier_only_p10 0.847, outlier_p10 0.821, gaussian 0.841, student_t_nu3 0.840, censoring 1.000, outlier_only_p05 0.883, outlier_p05 0.839, grouped_wider 0.818, laplace 0.850, outlier_only_p01 0.869, outlier_p01 0.827, student_t_nu5 0.836, student_t_nu10 0.801, shift_only_p05 0.972, shift_only_p10 0.946, sparse_dprime_p02 0.972, sparse_dprime_p05 0.955, sparse_dprime_p10 0.930, sparse_dprime_up_p02 0.972, sparse_dprime_up_p05 0.954, sparse_dprime_up_p10 0.929

## Level position 2

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.489, outlier_only_p10 0.722, outlier_p10 0.580, gaussian 0.592, student_t_nu3 0.643, censoring 1.000, outlier_only_p05 0.713, outlier_p05 0.638, grouped_wider 0.586, laplace 0.646, outlier_only_p01 0.692, outlier_p01 0.584, student_t_nu5 0.607, student_t_nu10 0.525, shift_only_p05 0.926, shift_only_p10 0.873, sparse_dprime_p02 0.938, sparse_dprime_p05 0.903, sparse_dprime_p10 0.820, sparse_dprime_up_p02 0.940, sparse_dprime_up_p05 0.876, sparse_dprime_up_p10 0.839


## Drop table

```
kind,dnn_graph_kernel_L0,dnn_graph_kernel_L1,dnn_graph_kernel_L2,lightgbm_mhg_gnn_L0,lightgbm_mhg_gnn_L1,lightgbm_mhg_gnn_L2,rf_ecfp4_L0,rf_ecfp4_L1,rf_ecfp4_L2,svm_pdv_L0,svm_pdv_L1,svm_pdv_L2
grouped_shifted,0.0208,0.1030,0.2830,0.0151,0.0783,0.2989,0.0067,0.0485,0.1828,0.0102,0.0585,0.1605
outlier_only_p10,0.0196,0.0919,0.1228,0.0189,0.0896,0.3383,-0.0018,0.0334,0.1327,0.0046,0.0115,0.0119
outlier_p10,0.0239,0.0893,0.1771,0.0116,0.0985,0.3140,0.0023,0.0419,0.1422,0.0113,0.0308,0.0804
gaussian,0.0187,0.0704,0.1756,0.0067,0.0691,0.2605,0.0005,0.0299,0.1391,0.0068,0.0445,0.1135
student_t_nu3,0.0180,0.0586,0.1451,0.0149,0.0751,0.2499,-0.0006,0.0275,0.1258,0.0031,0.0240,0.0659
censoring,0.0088,0.0751,0.1561,0.0327,0.1118,0.1974,0.0393,0.1063,0.1830,0.0228,0.0769,0.1586
outlier_only_p05,0.0136,0.0650,0.1482,0.0043,0.0528,0.2427,-0.0011,0.0179,0.0892,0.0006,0.0029,0.0030
outlier_p05,0.0151,0.0654,0.1557,0.0265,0.0923,0.2788,-0.0031,0.0331,0.1391,0.0027,0.0277,0.0903
grouped_wider,0.0345,0.0824,0.1636,0.0210,0.0951,0.3459,0.0067,0.0457,0.1723,0.0053,0.0310,0.0830
laplace,0.0332,0.0953,0.2037,0.0129,0.0799,0.2904,-0.0006,0.0264,0.1398,0.0080,0.0305,0.0742
outlier_only_p01,0.0214,0.0613,0.1619,0.0142,0.0457,0.1786,0.0059,0.0479,0.1929,0.0005,0.0006,0.0006
outlier_p01,0.0230,0.0631,0.1664,0.0025,0.0795,0.2694,0.0078,0.0426,0.1545,0.0120,0.0465,0.1169
student_t_nu5,0.0280,0.0807,0.1620,0.0141,0.0836,0.2988,0.0042,0.0452,0.1623,0.0070,0.0372,0.1005
student_t_nu10,0.0168,0.0856,0.1468,0.0090,0.0859,0.2951,0.0049,0.0436,0.1617,0.0053,0.0367,0.0974
shift_only_p05,0.0009,0.0070,0.0482,0.0055,0.0128,0.0509,-0.0023,-0.0003,0.0155,0.0034,0.0057,0.0067
shift_only_p10,-0.0039,0.0169,0.0814,0.0093,0.0269,0.0912,-0.0022,0.0076,0.0409,0.0034,0.0130,0.0263
sparse_dprime_p02,0.0002,0.0131,0.0334,0.0002,0.0019,0.0201,-0.0047,0.0021,0.0107,0.0019,0.0028,0.0031
sparse_dprime_p05,0.0024,0.0270,0.0512,-0.0024,0.0147,0.0685,0.0015,0.0079,0.0246,0.0039,0.0057,0.0066
sparse_dprime_p10,0.0155,0.0408,0.0958,0.0083,0.0329,0.1173,0.0022,0.0194,0.0543,0.0049,0.0075,0.0108
sparse_dprime_up_p02,-0.0009,0.0169,0.0331,-0.0035,-0.0008,0.0201,-0.0028,0.0017,0.0095,-0.0017,-0.0010,-0.0007
sparse_dprime_up_p05,0.0095,0.0311,0.0857,-0.0013,0.0153,0.0784,0.0004,0.0090,0.0397,-0.0007,0.0018,0.0029
sparse_dprime_up_p10,0.0164,0.0439,0.1203,0.0096,0.0408,0.1588,-0.0045,0.0111,0.0683,0.0026,0.0078,0.0112
```
