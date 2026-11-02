# f37 logd

Metric rmse (lower is better). Drop table: mean over folds and draws of damaged score minus clean score, sign flipped when higher is better, so positive = worse.

Correlations: Pearson between held-out predictions of two kinds on the same 5028 molecules, mean over the pairs. Diagonal = same kind, draw 0 vs 1.

## Level position 0

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.961, outlier_only_p10 0.973, outlier_p10 0.971, gaussian 0.968, student_t_nu3 0.971, censoring 1.000, outlier_only_p05 0.969, outlier_p05 0.970, grouped_wider 0.971, laplace 0.970, outlier_only_p01 0.975, outlier_p01 0.971, student_t_nu5 0.968, student_t_nu10 0.970, shift_only_p05 0.990, shift_only_p10 0.987, sparse_dprime_p02 0.992, sparse_dprime_p05 0.989, sparse_dprime_p10 0.985, sparse_dprime_up_p02 0.992, sparse_dprime_up_p05 0.990, sparse_dprime_up_p10 0.986

## Level position 1

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.844, outlier_only_p10 0.911, outlier_p10 0.896, gaussian 0.898, student_t_nu3 0.906, censoring 1.000, outlier_only_p05 0.892, outlier_p05 0.898, grouped_wider 0.902, laplace 0.902, outlier_only_p01 0.921, outlier_p01 0.897, student_t_nu5 0.894, student_t_nu10 0.899, shift_only_p05 0.981, shift_only_p10 0.972, sparse_dprime_p02 0.987, sparse_dprime_p05 0.979, sparse_dprime_p10 0.967, sparse_dprime_up_p02 0.988, sparse_dprime_up_p05 0.980, sparse_dprime_up_p10 0.967

## Level position 2

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.610, outlier_only_p10 0.797, outlier_p10 0.739, gaussian 0.739, student_t_nu3 0.791, censoring 1.000, outlier_only_p05 0.769, outlier_p05 0.753, grouped_wider 0.778, laplace 0.767, outlier_only_p01 0.816, outlier_p01 0.763, student_t_nu5 0.756, student_t_nu10 0.768, shift_only_p05 0.956, shift_only_p10 0.931, sparse_dprime_p02 0.974, sparse_dprime_p05 0.952, sparse_dprime_p10 0.917, sparse_dprime_up_p02 0.974, sparse_dprime_up_p05 0.950, sparse_dprime_up_p10 0.915


## Drop table

```
kind,dnn_chembert_L0,dnn_chembert_L1,dnn_chembert_L2,lightgbm_mhg_gnn_L0,lightgbm_mhg_gnn_L1,lightgbm_mhg_gnn_L2,rf_ecfp4_L0,rf_ecfp4_L1,rf_ecfp4_L2,svm_mordred_L0,svm_mordred_L1,svm_mordred_L2
grouped_shifted,0.0380,0.1949,0.4479,0.0324,0.1461,0.4253,0.0152,0.0819,0.2822,0.0254,0.1153,0.2733
outlier_only_p10,0.0352,0.1407,0.2302,0.0247,0.1076,0.3395,0.0050,0.0438,0.1839,0.0075,0.0109,0.0131
outlier_p10,0.0329,0.1418,0.2488,0.0221,0.1040,0.3286,0.0063,0.0527,0.1965,0.0192,0.0717,0.1587
gaussian,0.0399,0.1396,0.2222,0.0171,0.1047,0.3349,0.0083,0.0508,0.2054,0.0211,0.0815,0.1804
student_t_nu3,0.0404,0.1450,0.2306,0.0240,0.1091,0.3435,0.0058,0.0488,0.1809,0.0153,0.0538,0.1175
censoring,0.0211,0.1097,0.2433,0.0228,0.1149,0.2664,0.0183,0.0831,0.1973,0.0173,0.1021,0.2467
outlier_only_p05,0.0456,0.1526,0.2251,0.0214,0.1147,0.3611,0.0110,0.0705,0.2544,0.0053,0.0072,0.0078
outlier_p05,0.0319,0.1362,0.2497,0.0226,0.1141,0.3359,0.0061,0.0489,0.1992,0.0183,0.0714,0.1624
grouped_wider,0.0325,0.1380,0.2406,0.0231,0.1128,0.3359,0.0071,0.0481,0.1947,0.0187,0.0662,0.1394
laplace,0.0395,0.1445,0.2342,0.0243,0.1110,0.3402,0.0085,0.0563,0.1989,0.0169,0.0611,0.1341
outlier_only_p01,0.0365,0.1089,0.1807,0.0181,0.0737,0.2558,0.0048,0.0573,0.2087,0.0011,0.0012,0.0012
outlier_p01,0.0345,0.1417,0.2327,0.0239,0.1031,0.3263,0.0068,0.0505,0.1983,0.0191,0.0829,0.1863
student_t_nu5,0.0397,0.1462,0.2397,0.0314,0.1163,0.3491,0.0133,0.0614,0.2230,0.0235,0.0811,0.1730
student_t_nu10,0.0446,0.1477,0.2229,0.0258,0.1090,0.3384,0.0123,0.0598,0.2140,0.0234,0.0784,0.1663
shift_only_p05,0.0045,0.0207,0.0820,0.0077,0.0184,0.0568,-0.0016,0.0036,0.0207,0.0051,0.0085,0.0110
shift_only_p10,0.0145,0.0466,0.1595,0.0127,0.0311,0.1203,0.0020,0.0105,0.0557,0.0107,0.0191,0.0264
sparse_dprime_p02,0.0065,0.0137,0.0478,0.0043,0.0071,0.0266,0.0021,0.0019,0.0090,0.0008,0.0020,0.0039
sparse_dprime_p05,0.0115,0.0218,0.0673,0.0069,0.0186,0.0455,-0.0002,0.0022,0.0160,0.0041,0.0069,0.0078
sparse_dprime_p10,0.0154,0.0431,0.1221,0.0052,0.0203,0.0819,0.0035,0.0086,0.0370,0.0050,0.0111,0.0152
sparse_dprime_up_p02,0.0076,0.0156,0.0420,0.0028,0.0095,0.0253,0.0011,0.0019,0.0051,0.0013,0.0024,0.0051
sparse_dprime_up_p05,0.0125,0.0288,0.0918,0.0025,0.0130,0.0639,-0.0007,0.0005,0.0207,0.0043,0.0084,0.0100
sparse_dprime_up_p10,0.0152,0.0476,0.1518,0.0135,0.0413,0.1391,0.0003,0.0181,0.0653,0.0091,0.0186,0.0244
```
