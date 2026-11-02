# f37 caco2

Metric rmse (lower is better). Drop table: mean over folds and draws of damaged score minus clean score, sign flipped when higher is better, so positive = worse.

Correlations: Pearson between held-out predictions of two kinds on the same 2158 molecules, mean over the pairs. Diagonal = same kind, draw 0 vs 1.

## Level position 0

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.942, outlier_only_p10 0.948, outlier_p10 0.941, gaussian 0.945, student_t_nu3 0.949, censoring 1.000, outlier_only_p05 0.952, outlier_p05 0.943, grouped_wider 0.941, laplace 0.941, outlier_only_p01 0.958, outlier_p01 0.938, student_t_nu5 0.947, student_t_nu10 0.944, shift_only_p05 0.979, shift_only_p10 0.975, sparse_dprime_p02 0.980, sparse_dprime_p05 0.972, sparse_dprime_p10 0.958, sparse_dprime_up_p02 0.981, sparse_dprime_up_p05 0.974, sparse_dprime_up_p10 0.960

## Level position 1

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.789, outlier_only_p10 0.875, outlier_p10 0.841, gaussian 0.820, student_t_nu3 0.856, censoring 1.000, outlier_only_p05 0.867, outlier_p05 0.830, grouped_wider 0.830, laplace 0.833, outlier_only_p01 0.860, outlier_p01 0.829, student_t_nu5 0.842, student_t_nu10 0.843, shift_only_p05 0.964, shift_only_p10 0.951, sparse_dprime_p02 0.969, sparse_dprime_p05 0.944, sparse_dprime_p10 0.910, sparse_dprime_up_p02 0.968, sparse_dprime_up_p05 0.946, sparse_dprime_up_p10 0.920

## Level position 2

Same kind, draw 0 vs 1 (yardstick): grouped_shifted 0.527, outlier_only_p10 0.749, outlier_p10 0.677, gaussian 0.626, student_t_nu3 0.688, censoring 1.000, outlier_only_p05 0.705, outlier_p05 0.644, grouped_wider 0.623, laplace 0.590, outlier_only_p01 0.736, outlier_p01 0.655, student_t_nu5 0.656, student_t_nu10 0.667, shift_only_p05 0.926, shift_only_p10 0.884, sparse_dprime_p02 0.933, sparse_dprime_p05 0.878, sparse_dprime_p10 0.792, sparse_dprime_up_p02 0.933, sparse_dprime_up_p05 0.886, sparse_dprime_up_p10 0.832


## Drop table

```
kind,dnn_graph_kernel_L0,dnn_graph_kernel_L1,dnn_graph_kernel_L2,lightgbm_mhg_gnn_L0,lightgbm_mhg_gnn_L1,lightgbm_mhg_gnn_L2,rf_ecfp4_L0,rf_ecfp4_L1,rf_ecfp4_L2,svm_pdv_L0,svm_pdv_L1,svm_pdv_L2
grouped_shifted,0.0116,0.0550,0.1196,0.0099,0.0477,0.1462,0.0053,0.0223,0.0805,0.0060,0.0346,0.0963
outlier_only_p10,0.0096,0.0335,0.0625,0.0082,0.0393,0.1298,0.0025,0.0156,0.0593,0.0030,0.0056,0.0083
outlier_p10,0.0101,0.0432,0.0660,0.0061,0.0369,0.1319,0.0033,0.0172,0.0615,0.0041,0.0200,0.0553
gaussian,0.0128,0.0377,0.0730,0.0086,0.0454,0.1362,0.0033,0.0188,0.0703,0.0066,0.0317,0.0799
student_t_nu3,0.0047,0.0297,0.0657,0.0072,0.0333,0.1127,0.0023,0.0160,0.0558,0.0018,0.0121,0.0392
censoring,0.0112,0.0649,0.1184,0.0124,0.0647,0.1195,0.0084,0.0606,0.1109,0.0063,0.0673,0.1351
outlier_only_p05,0.0092,0.0361,0.0670,0.0053,0.0345,0.1174,0.0015,0.0150,0.0618,0.0017,0.0024,0.0031
outlier_p05,0.0115,0.0395,0.0585,0.0052,0.0342,0.1230,0.0024,0.0193,0.0701,0.0037,0.0198,0.0574
grouped_wider,0.0101,0.0350,0.0651,0.0051,0.0347,0.1179,0.0020,0.0161,0.0649,0.0027,0.0167,0.0495
laplace,0.0151,0.0400,0.0775,0.0077,0.0429,0.1342,0.0029,0.0181,0.0663,0.0057,0.0221,0.0547
outlier_only_p01,0.0097,0.0460,0.0543,0.0063,0.0265,0.0960,0.0019,0.0185,0.0517,0.0012,0.0012,0.0012
outlier_p01,0.0140,0.0407,0.0648,0.0075,0.0408,0.1336,0.0014,0.0184,0.0642,0.0043,0.0241,0.0733
student_t_nu5,0.0059,0.0501,0.0677,0.0092,0.0375,0.1321,0.0016,0.0163,0.0671,0.0033,0.0218,0.0599
student_t_nu10,0.0099,0.0428,0.0648,0.0072,0.0409,0.1313,0.0034,0.0150,0.0615,0.0047,0.0220,0.0651
shift_only_p05,0.0020,0.0042,0.0227,0.0004,0.0042,0.0208,0.0012,0.0014,0.0075,0.0006,0.0028,0.0053
shift_only_p10,0.0040,0.0093,0.0380,0.0028,0.0148,0.0491,0.0005,0.0053,0.0175,0.0021,0.0065,0.0149
sparse_dprime_p02,0.0019,0.0077,0.0161,0.0028,0.0030,0.0149,0.0006,0.0034,0.0047,-0.0002,0.0003,0.0005
sparse_dprime_p05,0.0056,0.0153,0.0345,0.0064,0.0135,0.0373,0.0018,0.0042,0.0137,0.0022,0.0039,0.0061
sparse_dprime_p10,0.0060,0.0232,0.0512,0.0047,0.0169,0.0590,0.0000,0.0051,0.0221,0.0018,0.0057,0.0082
sparse_dprime_up_p02,0.0018,0.0059,0.0186,0.0020,0.0056,0.0175,0.0009,0.0014,0.0043,0.0001,0.0007,0.0020
sparse_dprime_up_p05,0.0044,0.0135,0.0383,0.0019,0.0107,0.0402,0.0001,0.0035,0.0152,0.0002,0.0023,0.0035
sparse_dprime_up_p10,0.0130,0.0263,0.0664,0.0081,0.0235,0.0801,0.0016,0.0099,0.0380,0.0017,0.0063,0.0110
```
