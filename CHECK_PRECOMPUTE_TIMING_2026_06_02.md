# check_precompute_status.py --timing output, 2026-06-02

Captured from the user's terminal paste on 2026-06-02. This is **measured ground truth** for per-(dataset, condition, model) wall time at the time the corresponding cache file was written. Use this to build the per-config cost regression.

Format per condition:
- `[time] tier/strategy: n=N TOTALh (SECONDs)  n_configs=COUNT  written=DATE`
- Then per-model breakdown: `model TOTALh (SECONDs)  n_cfg=COUNT  per_cfg=Xs  share=Y%`

If a tier/strategy is missing from a dataset, that condition has not yet been computed for that dataset.

---

## ANTIBODY

### abagym_Ang2 (DONE 2/2)
```
[time] general/none: n=397 1.25h (4487s)  n_configs=3846  written=2026-05-08 13:42
    gp           0.49h (   1773s)  n_cfg=498   per_cfg=  3.56s  share= 39.5%
    xgboost      0.16h (    565s)  n_cfg=546   per_cfg=  1.04s  share= 12.6%
    mlp          0.16h (    560s)  n_cfg=468   per_cfg=  1.20s  share= 12.5%
    rf           0.13h (    452s)  n_cfg=420   per_cfg=  1.08s  share= 10.1%
    ridge        0.09h (    327s)  n_cfg=504   per_cfg=  0.65s  share=  7.3%
    knn          0.08h (    289s)  n_cfg=552   per_cfg=  0.52s  share=  6.4%
    lightgbm     0.08h (    285s)  n_cfg=432   per_cfg=  0.66s  share=  6.4%
    svm          0.07h (    235s)  n_cfg=426   per_cfg=  0.55s  share=  5.2%
[time] top/none: n=397 0.91h (3285s)  n_configs=5184  written=2026-06-01 12:23
    gp           0.26h (    929s)  n_cfg=720   per_cfg=  1.29s  share= 28.3%
    lightgbm     0.23h (    835s)  n_cfg=576   per_cfg=  1.45s  share= 25.4%
    xgboost      0.10h (    353s)  n_cfg=720   per_cfg=  0.49s  share= 10.8%
    knn          0.08h (    281s)  n_cfg=720   per_cfg=  0.39s  share=  8.6%
    rf           0.07h (    265s)  n_cfg=576   per_cfg=  0.46s  share=  8.1%
    mlp          0.07h (    249s)  n_cfg=576   per_cfg=  0.43s  share=  7.6%
    ridge        0.06h (    225s)  n_cfg=720   per_cfg=  0.31s  share=  6.8%
    svm          0.04h (    148s)  n_cfg=576   per_cfg=  0.26s  share=  4.5%
```

### abagym_EGFR (DONE 2/2)
```
[time] general/none: n=659 2.96h (10640s)  n_configs=4445  written=2026-05-08 19:04
    gp           1.44h (   5167s)  n_cfg=630   per_cfg=  8.20s  share= 48.6%
    xgboost      0.36h (   1279s)  n_cfg=616   per_cfg=  2.08s  share= 12.0%
    mlp          0.29h (   1045s)  n_cfg=532   per_cfg=  1.96s  share=  9.8%
    lightgbm     0.19h (    686s)  n_cfg=490   per_cfg=  1.40s  share=  6.4%
    rf           0.19h (    684s)  n_cfg=455   per_cfg=  1.50s  share=  6.4%
    ridge        0.18h (    645s)  n_cfg=588   per_cfg=  1.10s  share=  6.1%
    knn          0.17h (    598s)  n_cfg=630   per_cfg=  0.95s  share=  5.6%
    svm          0.15h (    536s)  n_cfg=504   per_cfg=  1.06s  share=  5.0%
[time] top/none: n=659 1.57h (5646s)  n_configs=4021  written=2026-05-30 15:39
    gp           0.95h (   3426s)  n_cfg=525   per_cfg=  6.53s  share= 60.7%
    xgboost      0.20h (    713s)  n_cfg=609   per_cfg=  1.17s  share= 12.6%
    mlp          0.13h (    483s)  n_cfg=462   per_cfg=  1.05s  share=  8.6%
    rf           0.09h (    314s)  n_cfg=476   per_cfg=  0.66s  share=  5.6%
    lightgbm     0.07h (    255s)  n_cfg=455   per_cfg=  0.56s  share=  4.5%
    ridge        0.05h (    193s)  n_cfg=553   per_cfg=  0.35s  share=  3.4%
    knn          0.05h (    164s)  n_cfg=539   per_cfg=  0.30s  share=  2.9%
    svm          0.03h (     99s)  n_cfg=402   per_cfg=  0.25s  share=  1.7%
```

### abagym_HER2 (DONE 2/2)
```
[time] general/none: n=146 0.21h (756s)  n_configs=3310  written=2026-05-08 08:33
    gp 0.07h n_cfg=465  per_cfg=0.51s
    rf 0.05h n_cfg=355  per_cfg=0.52s
    mlp 0.03h n_cfg=385  per_cfg=0.27s
    xgboost 0.03h n_cfg=480  per_cfg=0.19s
    ridge 0.01h n_cfg=470  per_cfg=0.11s
    knn 0.01h n_cfg=455  per_cfg=0.07s
    lightgbm 0.01h n_cfg=330  per_cfg=0.10s
    svm 0.01h n_cfg=370  per_cfg=0.06s
[time] top/none: n=146 5.49h (19764s)  n_configs=4860  written=2026-06-02 04:27
    xgboost      3.28h (  11818s)  n_cfg=675   per_cfg= 17.51s  share= 59.8%
    lightgbm     1.92h (   6926s)  n_cfg=540   per_cfg= 12.83s  share= 35.0%
    gp           0.08h (    288s)  n_cfg=675   per_cfg=  0.43s  share=  1.5%
    knn          0.07h (    265s)  n_cfg=675   per_cfg=  0.39s  share=  1.3%
    mlp          0.05h (    178s)  n_cfg=540   per_cfg=  0.33s  share=  0.9%
    rf           0.04h (    153s)  n_cfg=540   per_cfg=  0.28s  share=  0.8%
    ridge        0.03h (    115s)  n_cfg=675   per_cfg=  0.17s  share=  0.6%
    svm          0.01h (     21s)  n_cfg=540   per_cfg=  0.04s  share=  0.1%
```

### abagym_VEGF (DONE 2/2)
```
[time] general/none: n=418 1.20h (4303s)  n_configs=3606  written=2026-05-08 13:37
    gp           0.48h (   1742s)  n_cfg=474   per_cfg=  3.68s  share= 40.5%
    xgboost      0.16h (    558s)  n_cfg=534   per_cfg=  1.05s  share= 13.0%
    mlp          0.15h (    548s)  n_cfg=408   per_cfg=  1.34s  share= 12.7%
    rf           0.12h (    419s)  n_cfg=414   per_cfg=  1.01s  share=  9.7%
    ridge        0.09h (    328s)  n_cfg=510   per_cfg=  0.64s  share=  7.6%
    lightgbm     0.08h (    274s)  n_cfg=408   per_cfg=  0.67s  share=  6.4%
    knn          0.07h (    240s)  n_cfg=468   per_cfg=  0.51s  share=  5.6%
    svm          0.05h (    194s)  n_cfg=390   per_cfg=  0.50s  share=  4.5%
[time] top/none: n=418 0.82h (2939s)  n_configs=5832  written=2026-06-01 13:14
    gp           0.32h (   1145s)  n_cfg=810   per_cfg=  1.41s  share= 39.0%
    xgboost      0.11h (    384s)  n_cfg=810   per_cfg=  0.47s  share= 13.1%
    mlp          0.09h (    331s)  n_cfg=648   per_cfg=  0.51s  share= 11.3%
    knn          0.07h (    264s)  n_cfg=810   per_cfg=  0.33s  share=  9.0%
    rf           0.07h (    253s)  n_cfg=648   per_cfg=  0.39s  share=  8.6%
    ridge        0.07h (    253s)  n_cfg=810   per_cfg=  0.31s  share=  8.6%
    lightgbm     0.05h (    185s)  n_cfg=648   per_cfg=  0.29s  share=  6.3%
    svm          0.03h (    123s)  n_cfg=648   per_cfg=  0.19s  share=  4.2%
```

### abagym_lysozyme (DONE 3/3)
```
[time] general/none: n=627 0.52h (1874s)  n_configs=3745  written=2026-05-14 03:30
    gp           0.33h (   1189s)  n_cfg=448   per_cfg=  2.65s  share= 63.4%
    mlp          0.06h (    211s)  n_cfg=434   per_cfg=  0.49s  share= 11.2%
    rf           0.04h (    150s)  n_cfg=469   per_cfg=  0.32s  share=  8.0%
    xgboost      0.02h (     78s)  n_cfg=518   per_cfg=  0.15s  share=  4.2%
    lightgbm     0.02h (     63s)  n_cfg=434   per_cfg=  0.15s  share=  3.4%
    knn          0.02h (     62s)  n_cfg=567   per_cfg=  0.11s  share=  3.3%
    ridge        0.02h (     61s)  n_cfg=448   per_cfg=  0.14s  share=  3.3%
    svm          0.02h (     60s)  n_cfg=427   per_cfg=  0.14s  share=  3.2%
[time] general/uncapped: n=627 12.87h (46343s)  n_configs=3920  written=2026-05-14 19:17
    gp           8.73h (  31430s)  n_cfg=483   per_cfg= 65.07s  share= 67.8%
    mlp          1.45h (   5216s)  n_cfg=497   per_cfg= 10.50s  share= 11.3%
    rf           1.20h (   4334s)  n_cfg=462   per_cfg=  9.38s  share=  9.4%
    xgboost      0.48h (   1720s)  n_cfg=567   per_cfg=  3.03s  share=  3.7%
    ridge        0.38h (   1366s)  n_cfg=434   per_cfg=  3.15s  share=  2.9%
    svm          0.24h (    881s)  n_cfg=455   per_cfg=  1.94s  share=  1.9%
    lightgbm     0.23h (    846s)  n_cfg=434   per_cfg=  1.95s  share=  1.8%
    knn          0.15h (    550s)  n_cfg=588   per_cfg=  0.93s  share=  1.2%
[time] top/uncapped: n=627 18.21h (65540s)  n_configs=5292  written=2026-05-28 00:13
    gp          13.16h (  47391s)  n_cfg=735   per_cfg= 64.48s  share= 72.3%
    mlp          1.59h (   5707s)  n_cfg=588   per_cfg=  9.71s  share=  8.7%
    rf           1.47h (   5305s)  n_cfg=588   per_cfg=  9.02s  share=  8.1%
    ridge        0.59h (   2131s)  n_cfg=735   per_cfg=  2.90s  share=  3.3%
    xgboost      0.58h (   2083s)  n_cfg=735   per_cfg=  2.83s  share=  3.2%
    svm          0.31h (   1126s)  n_cfg=588   per_cfg=  1.91s  share=  1.7%
    lightgbm     0.31h (   1115s)  n_cfg=588   per_cfg=  1.90s  share=  1.7%
    knn          0.19h (    683s)  n_cfg=735   per_cfg=  0.93s  share=  1.0%
```

### abcov_kd (DONE 2/2)
```
[time] general/none: n=141 0.51h (1850s)  n_configs=3425  written=2026-05-08 09:27
    mlp 0.15h n_cfg=370  per_cfg=1.48s
    gp 0.10h n_cfg=455  per_cfg=0.76s
    rf 0.08h n_cfg=370  per_cfg=0.73s
    xgboost 0.06h n_cfg=505  per_cfg=0.43s
    ridge 0.04h n_cfg=475  per_cfg=0.29s
    knn 0.03h n_cfg=480  per_cfg=0.26s
    lightgbm 0.03h n_cfg=375  per_cfg=0.30s
    svm 0.03h n_cfg=395  per_cfg=0.24s
[time] top/none: n=141 0.24h (875s)  n_configs=4860  written=2026-06-01 08:24
    mlp 0.06h n_cfg=540 gp 0.04h n_cfg=675 rf 0.04h n_cfg=540 xgboost 0.03h n_cfg=675 ridge 0.02h n_cfg=675 knn 0.02h n_cfg=675 lightgbm 0.02h n_cfg=540 svm 0.01h n_cfg=540
```

### ddg_combined (DONE 4/4)
```
[time] general/none: n=429 1.10h (3960s)  n_configs=3282  written=2026-05-08 23:55
    gp           0.54h (   1943s)  n_cfg=444   per_cfg=  4.38s  share= 49.1%
    xgboost      0.16h (    564s)  n_cfg=510   per_cfg=  1.11s  share= 14.2%
    rf           0.09h (    316s)  n_cfg=348   per_cfg=  0.91s  share=  8.0%
    mlp          0.09h (    313s)  n_cfg=342   per_cfg=  0.92s  share=  7.9%
    ridge        0.07h (    235s)  n_cfg=456   per_cfg=  0.52s  share=  5.9%
    lightgbm     0.06h (    231s)  n_cfg=378   per_cfg=  0.61s  share=  5.8%
    knn          0.05h (    183s)  n_cfg=438   per_cfg=  0.42s  share=  4.6%
    svm          0.05h (    174s)  n_cfg=366   per_cfg=  0.48s  share=  4.4%
[time] general/uncapped: n=429 7.61h (27397s)  n_configs=5022  written=2026-05-12 11:35
    rf           1.88h (   6770s)  n_cfg=648   per_cfg= 10.45s  share= 24.7%
    xgboost      1.84h (   6614s)  n_cfg=810   per_cfg=  8.17s  share= 24.1%
    mlp          1.26h (   4545s)  n_cfg=648   per_cfg=  7.01s  share= 16.6%
    ridge        1.00h (   3593s)  n_cfg=810   per_cfg=  4.44s  share= 13.1%
    lightgbm     0.77h (   2789s)  n_cfg=648   per_cfg=  4.30s  share= 10.2%
    svm          0.50h (   1811s)  n_cfg=648   per_cfg=  2.80s  share=  6.6%
    knn          0.35h (   1274s)  n_cfg=810   per_cfg=  1.57s  share=  4.7%
[time] top/none: n=429 0.56h (2024s)  n_configs=3012  written=2026-05-26 09:53
    gp 0.34h xgboost 0.08h rf 0.04h mlp 0.04h lightgbm 0.02h ridge 0.02h svm 0.01h knn 0.01h
[time] top/uncapped: n=429 4.29h (15457s)  n_configs=2622  written=2026-05-26 15:13
    xgboost 1.18h rf 1.09h mlp 0.70h ridge 0.58h lightgbm 0.38h svm 0.23h knn 0.13h
```

### flab_aggregation_jain (DONE 5/5)
```
[time] general/none: n=137 0.44h (1588s)  n_configs=4320  written=2026-05-11 04:22
[time] general/uncapped: n=137 3.75h (13484s)  n_configs=3095  written=2026-05-11 03:57
    gp 1.02h rf 0.90h mlp 0.89h xgboost 0.43h lightgbm 0.16h ridge 0.16h knn 0.12h svm 0.07h
[time] top/none: n=137 0.21h (758s)  n_configs=4860  written=2026-06-01 15:34
[time] top/uncapped: n=137 4.36h (15683s)  n_configs=4860  written=2026-06-01 17:33
    rf 1.11h gp 1.03h mlp 0.92h xgboost 0.49h lightgbm 0.27h ridge 0.23h knn 0.18h svm 0.11h
[time] top/capped: n=137 35.21h (126763s)  n_configs=2715  written=2026-05-12 22:47
    rf          20.37h (  73322s)  n_cfg=290   per_cfg=252.83s  share= 57.8%
    mlp          3.45h (  12416s)  n_cfg=320   per_cfg= 38.80s  share=  9.8%
    knn          2.52h (   9064s)  n_cfg=360   per_cfg= 25.18s  share=  7.2%
    gp           2.49h (   8954s)  n_cfg=340   per_cfg= 26.34s  share=  7.1%
    xgboost      1.89h (   6819s)  n_cfg=375   per_cfg= 18.18s  share=  5.4%
    svm          1.66h (   5969s)  n_cfg=330   per_cfg= 18.09s  share=  4.7%
    lightgbm     1.63h (   5866s)  n_cfg=325   per_cfg= 18.05s  share=  4.6%
    ridge        1.21h (   4353s)  n_cfg=375   per_cfg= 11.61s  share=  3.4%
```

### flab_aggregation_jain_hicrt (DONE 5/5)
```
[time] general/none: n=137 0.44h (1569s)  n_configs=2855  written=2026-05-08 22:56
[time] general/uncapped: n=137 7.27h (26177s)  n_configs=3255  written=2026-05-09 08:19
    mlp 2.78h gp 2.37h rf 0.99h xgboost 0.41h ridge 0.22h lightgbm 0.21h knn 0.17h svm 0.12h
[time] top/none: n=137 0.16h (586s)  n_configs=2835  written=2026-05-13 18:12
[time] top/uncapped: n=137 6.02h (21680s)  n_configs=2775  written=2026-05-14 01:46
[time] top/capped: n=137 103.99h (374379s)  n_configs=3780  written=2026-05-17 06:04
    rf          64.96h ( 233850s)  n_cfg=420   per_cfg=556.79s  share= 62.5%
    mlp         12.29h (  44254s)  n_cfg=420   per_cfg=105.37s  share= 11.8%
    gp           7.72h (  27806s)  n_cfg=525   per_cfg= 52.96s  share=  7.4%
    knn          5.56h (  19999s)  n_cfg=525   per_cfg= 38.09s  share=  5.3%
    xgboost      3.63h (  13058s)  n_cfg=525   per_cfg= 24.87s  share=  3.5%
    lightgbm     3.39h (  12198s)  n_cfg=420   per_cfg= 29.04s  share=  3.3%
    svm          3.36h (  12095s)  n_cfg=420   per_cfg= 28.80s  share=  3.2%
    ridge        3.09h (  11119s)  n_cfg=525   per_cfg= 21.18s  share=  3.0%
```

### flab_binding_adams (DONE 4/4) — n=5000
```
[time] general/none: n=5000 21.32h (76760s)  n_configs=6480  written=2026-05-14 21:46
    mlp          4.25h (  15295s)  n_cfg=756   per_cfg= 20.23s  share= 19.9%
    ridge        3.35h (  12046s)  n_cfg=1080  per_cfg= 11.15s  share= 15.7%
    lightgbm     3.17h (  11418s)  n_cfg=864   per_cfg= 13.21s  share= 14.9%
    xgboost      2.91h (  10480s)  n_cfg=1080  per_cfg=  9.70s  share= 13.7%
    rf           2.88h (  10360s)  n_cfg=864   per_cfg= 11.99s  share= 13.5%
    knn          2.84h (  10230s)  n_cfg=1080  per_cfg=  9.47s  share= 13.3%
    svm          1.93h (   6931s)  n_cfg=756   per_cfg=  9.17s  share=  9.0%
[time] general/uncapped: n=5000 75.58h (272070s)  n_configs=3940  written=2026-05-18 19:59
    ridge       22.90h (  82438s)  n_cfg=663   per_cfg=124.34s  share= 30.3%
    mlp         17.65h (  63542s)  n_cfg=357   per_cfg=177.99s  share= 23.4%
    lightgbm    14.51h (  52251s)  n_cfg=544   per_cfg= 96.05s  share= 19.2%
    rf           8.85h (  31863s)  n_cfg=504   per_cfg= 63.22s  share= 11.7%
    xgboost      5.39h (  19419s)  n_cfg=720   per_cfg= 26.97s  share=  7.1%
    knn          4.09h (  14733s)  n_cfg=648   per_cfg= 22.74s  share=  5.4%
    svm          2.17h (   7824s)  n_cfg=504   per_cfg= 15.52s  share=  2.9%
[time] top/none: n=5000 8.76h (31518s)  n_configs=4838  written=2026-05-26 20:49
    mlp 1.89h ridge 1.39h xgboost 1.34h lightgbm 1.28h rf 1.19h knn 0.83h svm 0.83h
[time] top/uncapped: n=5000 58.14h (209287s)  n_configs=3842  written=2026-05-29 22:56
    ridge       21.18h (  76246s)  n_cfg=680   per_cfg=112.13s  share= 36.4%
    mlp         14.63h (  52673s)  n_cfg=378   per_cfg=139.35s  share= 25.2%
    lightgbm    10.40h (  37440s)  n_cfg=536   per_cfg= 69.85s  share= 17.9%
    rf           5.85h (  21056s)  n_cfg=480   per_cfg= 43.87s  share= 10.1%
    xgboost      3.13h (  11269s)  n_cfg=720   per_cfg= 15.65s  share=  5.4%
    knn          2.01h (   7231s)  n_cfg=544   per_cfg= 13.29s  share=  3.5%
    svm          0.94h (   3371s)  n_cfg=504   per_cfg=  6.69s  share=  1.6%
```

### flab_binding_koenig (partial 3/4, top/uncapped missing) — n=4275
```
[time] general/none: n=4275 10.24h (36851s)  n_configs=3347  written=2026-05-14 22:57
[time] general/uncapped: n=4275 151.77h (546385s)  n_configs=4307  written=2026-05-31 02:53
    svm         42.49h ( 152964s)  n_cfg=492   per_cfg=310.90s  share= 28.0%
    mlp         38.25h ( 137715s)  n_cfg=504   per_cfg=273.24s  share= 25.2%
    ridge       26.08h (  93874s)  n_cfg=719   per_cfg=130.56s  share= 17.2%
    xgboost     17.23h (  62021s)  n_cfg=720   per_cfg= 86.14s  share= 11.4%
    lightgbm    13.59h (  48923s)  n_cfg=576   per_cfg= 84.94s  share=  9.0%
    rf          10.17h (  36598s)  n_cfg=576   per_cfg= 63.54s  share=  6.7%
    knn          3.97h (  14290s)  n_cfg=720   per_cfg= 19.85s  share=  2.6%
[time] top/none: n=4275 5.73h (20614s)  n_configs=3676  written=2026-05-31 10:46
```

### flab_binding_phillips_h3 (DONE 4/4) — n=5000
```
[time] general/none: n=5000 25.44h (91577s)  n_configs=6479  written=2026-05-15 02:16
    mlp          5.86h (  21090s)  n_cfg=756   per_cfg= 27.90s  share= 23.0%
    xgboost      4.38h (  15762s)  n_cfg=1080  per_cfg= 14.59s  share= 17.2%
    ridge        3.59h (  12927s)  n_cfg=1079  per_cfg= 11.98s  share= 14.1%
    lightgbm     3.03h (  10899s)  n_cfg=864   per_cfg= 12.61s  share= 11.9%
    rf           2.99h (  10759s)  n_cfg=864   per_cfg= 12.45s  share= 11.7%
    knn          2.93h (  10552s)  n_cfg=1080  per_cfg=  9.77s  share= 11.5%
    svm          2.66h (   9588s)  n_cfg=756   per_cfg= 12.68s  share= 10.5%
[time] general/uncapped: n=5000 151.65h (545939s)  n_configs=4308  written=2026-05-31 10:01
[time] top/none: n=5000 7.17h (25822s)  n_configs=4016  written=2026-05-31 15:48
```

### flab_expression_adams (DONE 4/4) — n=5000
```
[time] general/none: n=5000 22.47h (80883s)  n_configs=6479  written=2026-05-14 23:16
[time] general/uncapped: n=5000 34.17h (123021s)  n_configs=3330  written=2026-05-16 15:02
[time] top/none: n=5000 8.12h (29239s)  n_configs=4142  written=2026-05-26 20:07
[time] top/uncapped: n=5000 25.09h (90328s)  n_configs=3687  written=2026-05-28 05:43
```

### flab_expression_koenig (partial 3/4, top/uncapped missing) — n=4275
```
[time] general/none: n=4275 9.57h (34441s)  n_configs=3361  written=2026-05-15 00:21
[time] general/uncapped: n=4275 110.97h (399496s)  n_configs=3700  written=2026-05-20 05:52
    svm         36.94h ( 132992s)  n_cfg=441   per_cfg=301.57s  share= 33.3%
[time] top/none: n=4275 5.70h (20530s)  n_configs=3753  written=2026-05-26 17:04
```

### flab_immunogenicity_marks (DONE 2/2)
```
[time] general/none: n=217 0.31h (1108s)  n_configs=3065  written=2026-05-08 10:40
[time] top/none: n=217 0.38h (1354s)  n_configs=4860  written=2026-06-01 09:44
```

### flab_pk_jain_fcrnrt (DONE 5/5) — n=132
```
[time] general/none: n=132 0.15h (548s)
[time] general/uncapped: n=132 3.91h (14083s)
    gp 1.77h rf 0.89h xgboost 0.37h mlp 0.31h lightgbm 0.19h ridge 0.17h knn 0.13h svm 0.09h
[time] top/none: n=132 0.11h (390s)
[time] top/uncapped: n=132 1.81h (6509s)
[time] top/capped: n=132 44.17h (158998s)
    rf          25.51h (  91835s)  n_cfg=275   per_cfg=333.95s  share= 57.8%
    gp           4.70h (  16932s)  n_cfg=370   per_cfg= 45.76s  share= 10.6%
    knn          3.44h (  12368s)  n_cfg=310   per_cfg= 39.90s  share=  7.8%
```

### flab_pk_jain_heprt (DONE 5/5) — n=130
```
[time] general/none: n=130 0.28h (1013s)
[time] general/uncapped: n=130 4.70h (16915s)
[time] top/none: n=130 0.18h (662s)
[time] top/uncapped: n=130 2.46h (8857s)
[time] top/capped: n=130 31.67h (114030s)
    rf 17.55h gp 3.21h knn 2.77h mlp 2.09h
```

### flab_polyreact_jain_bvp (DONE 2/2) — n=137
```
[time] general/none: n=137 0.35h (1273s)
[time] top/none: n=137 0.23h (835s)
```

### flab_polyreact_jain_cic (DONE 2/2) — n=137
```
[time] general/none: n=137 0.49h (1755s)
[time] top/none: n=137 0.41h (1465s)
```

### flab_thermo_jain (DONE 5/5) — n=137
```
[time] general/none: n=137 0.54h (1960s)
[time] general/uncapped: n=137 8.73h (31414s)
    mlp 5.32h gp 1.58h rf 0.82h xgboost 0.42h
[time] top/none: n=137 0.43h (1542s)
[time] top/uncapped: n=137 9.09h (32729s)
[time] top/capped: n=137 101.21h (364360s)
    rf          49.30h
    mlp         24.79h
    gp           7.20h
```

### flab_thermo_tresanco (DONE 4/4) — n=672
```
[time] general/none: n=672 5.57h (20042s)
[time] general/uncapped: n=672 14.56h (52423s)
[time] top/none: n=672 5.23h (18819s)
[time] top/uncapped: n=672 10.03h (36090s)
```

### ginkgo_acsins (DONE 5/5) — n=242
```
[time] general/none: n=242 0.56h (2024s)
[time] general/uncapped: n=242 4.64h (16711s)
[time] top/none: n=242 0.13h (454s)
[time] top/uncapped: n=242 3.27h (11788s)
[time] top/capped: n=242 81.18h (292232s)
    rf          50.31h
    gp           7.21h
    svm          5.34h
    mlp          5.14h
    knn          4.58h
```

### ginkgo_colloidal (DONE 5/5) — n=242
```
[time] general/none: n=242 0.58h
[time] general/uncapped: n=242 9.92h
[time] top/none: n=242 0.05h
[time] top/uncapped: n=242 3.40h
[time] top/capped: n=242 53.80h
    rf 31.61h gp 5.55h knn 4.57h
```

### ginkgo_heparin (DONE 5/5) — n=94
```
[time] general/none: n=94 0.26h
[time] general/uncapped: n=94 2.67h
[time] top/none: n=94 0.18h (written 2026-06-02 14:15 - VERY RECENT)
[time] top/uncapped: n=94 2.44h (written 2026-06-02 17:15 - VERY RECENT)
[time] top/capped: n=94 28.63h
    rf 18.03h mlp 2.51h knn 2.09h gp 1.98h
```

### ginkgo_hic (DONE 5/5) — n=242
```
[time] general/none: n=242 0.63h
[time] general/uncapped: n=242 5.77h
[time] top/none: n=242 0.17h
[time] top/uncapped: n=242 3.79h
[time] top/capped: n=242 71.96h
```

### ginkgo_pr_cho (DONE 5/5) — n=197
```
[time] general/none: n=197 0.73h
[time] general/uncapped: n=197 4.23h
[time] top/none: n=197 0.03h
[time] top/uncapped: n=197 2.20h
[time] top/capped: n=197 67.82h
```

### ginkgo_titer (DONE 5/5) — n=239
```
[time] general/none: n=239 0.83h
[time] general/uncapped: n=239 8.11h
[time] top/none: n=239 0.08h
[time] top/uncapped: n=239 3.64h
[time] top/capped: n=239 122.52h
    rf          82.36h
    mlp         14.09h
    gp           6.89h
```

### humanness_therapeutic (DONE 2/2) — n=564
```
[time] general/none: n=564 1.16h (4163s)  n_configs=3404  written=2026-05-30 10:57
[time] top/none: n=564 1.30h (4686s)  n_configs=6696  written=2026-06-01 12:51
```

### tz_her2_binary (DONE 2/2) — n=5000
```
[time] general/none: n=5000 12.85h (46257s)  n_configs=4127  written=2026-06-01 00:07
    svm          2.77h (   9961s)  n_cfg=480   per_cfg= 20.75s  share= 21.5%
    xgboost      2.11h (   7598s)  n_cfg=712   per_cfg= 10.67s  share= 16.4%
    knn          1.90h (   6830s)  n_cfg=784   per_cfg=  8.71s  share= 14.8%
    ridge        1.64h (   5906s)  n_cfg=592   per_cfg=  9.98s  share= 12.8%
    lightgbm     1.60h (   5752s)  n_cfg=552   per_cfg= 10.42s  share= 12.4%
    mlp          1.46h (   5250s)  n_cfg=511   per_cfg= 10.27s  share= 11.3%
    rf           1.38h (   4961s)  n_cfg=496   per_cfg= 10.00s  share= 10.7%
[time] top/none: n=5000 4.08h (14676s)  n_configs=4189  written=2026-05-31 11:36
```

### tz_her2_multiclass (DONE 2/2) — n=5000
```
[time] general/none: n=5000 16.32h (58762s)  n_configs=4021  written=2026-05-14 19:20
    svm          5.24h (  18871s)  n_cfg=490   per_cfg= 38.51s  share= 32.1%
[time] top/none: n=5000 13.34h (48030s)  n_configs=5038  written=2026-05-15 11:21
    svm          6.88h (  24785s)  n_cfg=588   per_cfg= 42.15s  share= 51.6%
```

---

## DTA

### bindingdb_kd (partial 1/4) — n=30000
```
[time] general/none: n=30000 96.23h (346419s)  n_configs=3868  written=2026-05-19 11:15
    ridge       19.78h (  71223s)  n_cfg=736   per_cfg= 96.77s  share= 20.6%
    knn         19.55h (  70381s)  n_cfg=600   per_cfg=117.30s  share= 20.3%
    rf          17.32h (  62353s)  n_cfg=664   per_cfg= 93.90s  share= 18.0%
    xgboost     14.66h (  52768s)  n_cfg=736   per_cfg= 71.70s  share= 15.2%
    mlp         13.35h (  48053s)  n_cfg=532   per_cfg= 90.32s  share= 13.9%
    lightgbm    11.57h (  41642s)  n_cfg=600   per_cfg= 69.40s  share= 12.0%
```

### kiba (partial 1/4) — n=30000
```
[time] general/none: n=30000 121.83h (438589s)  n_configs=4450  written=2026-05-20 10:38
    ridge       27.21h (  97952s)  n_cfg=838   per_cfg=116.89s  share= 22.3%
    knn         21.62h (  77842s)  n_cfg=840   per_cfg= 92.67s  share= 17.7%
    xgboost     21.19h (  76266s)  n_cfg=840   per_cfg= 90.79s  share= 17.4%
    mlp         18.69h (  67302s)  n_cfg=588   per_cfg=114.46s  share= 15.3%
    rf          17.29h (  62239s)  n_cfg=672   per_cfg= 92.62s  share= 14.2%
    lightgbm    15.83h (  56988s)  n_cfg=672   per_cfg= 84.80s  share= 13.0%
```

---

## MOLECULAR

### adme_hlm (DONE 4/4) — n=3086
```
[time] general/none: n=3086 7.97h (28703s)  n_configs=6480  written=2026-05-14 00:57
    xgboost      2.13h (   7665s)  n_cfg=1080  per_cfg=  7.10s  share= 26.7%
    svm          1.28h (   4608s)  n_cfg=756   per_cfg=  6.10s  share= 16.1%
    mlp          1.19h (   4294s)  n_cfg=756   per_cfg=  5.68s  share= 15.0%
    lightgbm     1.14h (   4120s)  n_cfg=864   per_cfg=  4.77s  share= 14.4%
    ridge        0.98h (   3546s)  n_cfg=1080  per_cfg=  3.28s  share= 12.4%
    rf           0.81h (   2915s)  n_cfg=864   per_cfg=  3.37s  share= 10.2%
    knn          0.43h (   1555s)  n_cfg=1080  per_cfg=  1.44s  share=  5.4%
[time] general/uncapped: n=3086 95.27h (342956s)  n_configs=3925  written=2026-05-19 00:16
    svm         27.85h ( 100265s)  n_cfg=504   per_cfg=198.94s  share= 29.2%
    xgboost     18.90h (  68040s)  n_cfg=616   per_cfg=110.46s  share= 19.8%
    ridge       17.02h (  61280s)  n_cfg=688   per_cfg= 89.07s  share= 17.9%
    lightgbm    15.56h (  56030s)  n_cfg=544   per_cfg=103.00s  share= 16.3%
    mlp          7.56h (  27231s)  n_cfg=413   per_cfg= 65.93s  share=  7.9%
    rf           6.18h (  22232s)  n_cfg=528   per_cfg= 42.11s  share=  6.5%
    knn          2.19h (   7878s)  n_cfg=632   per_cfg= 12.46s  share=  2.3%
[time] top/none: n=3086 4.57h (16442s)  n_configs=5473  written=2026-05-26 15:19
[time] top/uncapped: n=3086 75.00h (269985s)  n_configs=4248  written=2026-06-02 14:30
```

### adme_mdr1 (DONE 4/4) — n=2641
```
[time] general/none: n=2641 6.90h (24827s)  n_configs=6480
[time] general/uncapped: n=2641 80.40h (289447s)  n_configs=4000  written=2026-05-18 03:14
    svm         19.93h
    xgboost     16.67h
    ridge       16.16h
    lightgbm    14.75h
    rf           5.77h
    mlp          5.43h
    knn          1.69h
[time] top/none: n=2641 3.52h (12671s)  n_configs=4361
[time] top/uncapped: n=2641 72.57h (261268s)  n_configs=3932  written=2026-05-30 06:52
```

### adme_rlm (DONE 4/4) — n=3053
```
[time] general/none: n=3053 8.41h (30288s)  n_configs=6480
[time] general/uncapped: n=3053 102.71h (369741s)  n_configs=3742
    svm         29.96h
    xgboost     21.76h
    ridge       17.60h
    lightgbm    14.83h
    mlp         10.01h
    rf           6.29h
    knn          2.26h
[time] top/none: n=3053 4.23h (15213s)  n_configs=4702
[time] top/uncapped: needs to be computed (currently running)
```

### caco2 (DONE 4/4) — n=2158
```
[time] general/none: n=2158 4.53h (16302s)  n_configs=5984  written=2026-05-12 08:56
    xgboost      1.17h (   4223s)  n_cfg=992   per_cfg=  4.26s  share= 25.9%
    lightgbm     0.76h (   2736s)  n_cfg=816   per_cfg=  3.35s  share= 16.8%
    mlp          0.75h (   2711s)  n_cfg=651   per_cfg=  4.16s  share= 16.6%
    svm          0.57h (   2036s)  n_cfg=749   per_cfg=  2.72s  share= 12.5%
    rf           0.53h (   1919s)  n_cfg=848   per_cfg=  2.26s  share= 11.8%
    ridge        0.48h (   1736s)  n_cfg=928   per_cfg=  1.87s  share= 10.6%
    knn          0.26h (    942s)  n_cfg=1000  per_cfg=  0.94s  share=  5.8%
[time] general/uncapped: n=2158 63.31h (227909s)  n_configs=4320  written=2026-05-15 18:09
    xgboost     13.58h (  48892s)  n_cfg=720   per_cfg= 67.91s  share= 21.5%
    lightgbm    12.84h (  46232s)  n_cfg=576   per_cfg= 80.26s  share= 20.3%
    svm         11.49h (  41369s)  n_cfg=504   per_cfg= 82.08s  share= 18.2%
    ridge       11.39h (  41001s)  n_cfg=720   per_cfg= 56.95s  share= 18.0%
    mlp          7.11h (  25612s)  n_cfg=504   per_cfg= 50.82s  share= 11.2%
    rf           4.98h (  17930s)  n_cfg=576   per_cfg= 31.13s  share=  7.9%
    knn          1.91h (   6873s)  n_cfg=720   per_cfg=  9.55s  share=  3.0%
[time] top/none: n=2158 4.11h (14792s)  n_configs=6480  written=2026-05-26 10:11
[time] top/uncapped: n=2158 62.32h (224358s)  n_configs=4163  written=2026-05-29 14:12
```

### herg (partial 1/4) — n=7772
```
[time] general/none: n=7772 29.44h (105986s)  n_configs=5048  written=2026-05-15 06:36
    svm         19.31h (  69506s)  n_cfg=644   per_cfg=107.93s  share= 65.6%
    xgboost      2.44h (   8774s)  n_cfg=880   per_cfg=  9.97s  share=  8.3%
    rf           1.69h (   6093s)  n_cfg=728   per_cfg=  8.37s  share=  5.7%
    ridge        1.65h (   5945s)  n_cfg=656   per_cfg=  9.06s  share=  5.6%
    lightgbm     1.65h (   5926s)  n_cfg=704   per_cfg=  8.42s  share=  5.6%
    mlp          1.45h (   5205s)  n_cfg=532   per_cfg=  9.78s  share=  4.9%
    knn          1.26h (   4536s)  n_cfg=904   per_cfg=  5.02s  share=  4.3%
```

### herg_ki (DONE 4/4) — n=1412
```
[time] general/none: n=1412 3.47h (12499s)  n_configs=6480  written=2026-05-12 07:00
[time] general/uncapped: n=1412 47.97h (172709s)  n_configs=4320  written=2026-05-14 20:01
    xgboost     11.19h (  40271s)  n_cfg=720   per_cfg= 55.93s  share= 23.3%
    lightgbm    10.07h (  36246s)  n_cfg=576   per_cfg= 62.93s  share= 21.0%
    mlp         10.04h (  36150s)  n_cfg=504   per_cfg= 71.73s  share= 20.9%
    ridge        6.15h (  22155s)  n_cfg=720   per_cfg= 30.77s  share= 12.8%
    svm          5.87h (  21140s)  n_cfg=504   per_cfg= 41.94s  share= 12.2%
    rf           3.46h (  12447s)  n_cfg=576   per_cfg= 21.61s  share=  7.2%
    knn          1.19h (   4300s)  n_cfg=720   per_cfg=  5.97s  share=  2.5%
[time] top/none: n=1412 2.54h (9159s)  n_configs=4643  written=2026-05-26 08:11
[time] top/uncapped: n=1412 45.27h (162961s)  n_configs=3964  written=2026-05-28 16:52
```

### logd (partial 1/4) — n=5028
```
[time] general/none: n=5028 20.45h (73614s)  n_configs=6097  written=2026-05-14 18:44
    svm          5.86h (  21092s)  n_cfg=749   per_cfg= 28.16s  share= 28.7%
    mlp          3.86h (  13880s)  n_cfg=756   per_cfg= 18.36s  share= 18.9%
    xgboost      3.27h (  11787s)  n_cfg=960   per_cfg= 12.28s  share= 16.0%
    ridge        2.51h (   9047s)  n_cfg=1024  per_cfg=  8.84s  share= 12.3%
    lightgbm     1.97h (   7109s)  n_cfg=800   per_cfg=  8.89s  share=  9.7%
    rf           1.59h (   5736s)  n_cfg=752   per_cfg=  7.63s  share=  7.8%
    knn          1.38h (   4962s)  n_cfg=1056  per_cfg=  4.70s  share=  6.7%
```

### qm9 (partial 2/4) — n=9978
```
[time] general/none: n=9978 20.64h (74291s)  n_configs=4391  written=2026-05-15 01:44
    svm          7.39h (  26615s)  n_cfg=504   per_cfg= 52.81s  share= 35.8%
    mlp          3.91h (  14082s)  n_cfg=511   per_cfg= 27.56s  share= 19.0%
    xgboost      2.37h (   8524s)  n_cfg=728   per_cfg= 11.71s  share= 11.5%
    ridge        2.24h (   8049s)  n_cfg=760   per_cfg= 10.59s  share= 10.8%
    rf           1.82h (   6558s)  n_cfg=576   per_cfg= 11.39s  share=  8.8%
    lightgbm     1.70h (   6110s)  n_cfg=592   per_cfg= 10.32s  share=  8.2%
    knn          1.21h (   4352s)  n_cfg=720   per_cfg=  6.04s  share=  5.9%
[time] top/none: n=9978 23.81h (85731s)  n_configs=5952  written=2026-04-24 23:31
    svm         11.32h (  40745s)  n_cfg=768   per_cfg= 53.05s  share= 47.5%
    mlp          4.31h (  15525s)  n_cfg=768   per_cfg= 20.22s  share= 18.1%
    xgboost      2.17h (   7802s)  n_cfg=960   per_cfg=  8.13s  share=  9.1%
    ridge        1.98h (   7111s)  n_cfg=960   per_cfg=  7.41s  share=  8.3%
    rf           1.60h (   5751s)  n_cfg=768   per_cfg=  7.49s  share=  6.7%
    lightgbm     1.44h (   5183s)  n_cfg=768   per_cfg=  6.75s  share=  6.0%
    knn          1.00h (   3613s)  n_cfg=960   per_cfg=  3.76s  share=  4.2%
```

---

## NUCLEOTIDE

### nt_H3K4me3 (partial 3/4) — n=18244
```
[time] general/none: n=18244 4.63h (16662s)  n_configs=1563  written=2026-05-30 18:25
    rf 2.10h lightgbm 1.32h mlp 1.21h
[time] general/uncapped: n=18244 140.94h (507387s)  n_configs=3096  written=2026-05-20 04:30
    xgboost     40.37h ( 145350s)  n_cfg=720   per_cfg=201.87s  share= 28.6%
    rf          39.53h ( 142295s)  n_cfg=576   per_cfg=247.04s  share= 28.0%
    lightgbm    26.06h (  93799s)  n_cfg=576   per_cfg=162.85s  share= 18.5%
    mlp         22.88h (  82364s)  n_cfg=504   per_cfg=163.42s  share= 16.2%
    knn         12.11h (  43579s)  n_cfg=720   per_cfg= 60.53s  share=  8.6%
[time] top/none: but missing for this dataset per check_precompute
```

### nt_promoter_tata (partial 3/4) — n=5274
```
[time] general/none: n=5274 2.93h (10550s)  n_configs=1995  written=2026-05-30 14:15
    svm          1.47h (   5309s)  n_cfg=396   per_cfg= 13.41s  share= 50.3%
    lightgbm     0.61h (   2184s)  n_cfg=568   per_cfg=  3.84s  share= 20.7%
    rf           0.51h (   1846s)  n_cfg=576   per_cfg=  3.20s  share= 17.5%
    mlp          0.34h (   1211s)  n_cfg=455   per_cfg=  2.66s  share= 11.5%
[time] general/uncapped: n=5274 51.19h (184301s)  n_configs=2436  written=2026-06-02 09:06
    svm         28.09h ( 101120s)  n_cfg=504   per_cfg=200.64s  share= 54.9%
    lightgbm    11.95h (  43008s)  n_cfg=672   per_cfg= 64.00s  share= 23.3%
    rf           7.18h (  25844s)  n_cfg=672   per_cfg= 38.46s  share= 14.0%
    mlp          3.98h (  14329s)  n_cfg=588   per_cfg= 24.37s  share=  7.8%
[time] top/none: n=5274 0.58h (2098s)  n_configs=1392  written=2026-06-02 09:52
```

### gb_human_enhancers_cohn (partial 2/4) — n=27791
```
[time] general/none: n=27791 9.01h (32453s)  n_configs=2705  written=2026-05-12 23:37
    rf 2.77h xgboost 2.12h lightgbm 1.54h mlp 1.34h knn 1.24h
[time] general/uncapped: n=27791 106.41h (383060s)  n_configs=2575  written=2026-05-19 13:31
    rf          41.50h
    xgboost     29.67h
    lightgbm    17.65h
    mlp          9.91h
    knn          7.68h
```

### gb_demo_human_or_worm (partial 1/4) — n=100000
```
[time] general/none: n=100000 53.53h (192719s)  n_configs=3268  written=2026-05-18 15:34
    rf 15.41h knn 12.02h xgboost 10.63h mlp 9.30h lightgbm 6.17h
```

### Other nt_* general/none (partial 1/4 each, all measured)
```
nt_H2AFZ        general/none n=33000  10.26h
nt_H3K27ac      general/none n=31616  11.46h
nt_H3K27me3     general/none n=33000  8.16h
nt_H3K36me3     general/none n=33000  8.40h
nt_H3K4me1      general/none n=33000  10.64h
nt_H3K4me2      general/none n=32138  8.31h
nt_H3K9ac       general/none n=24278  9.02h
nt_H3K9me3      general/none n=28288  2.29h
nt_H4K20me1     general/none n=32270  5.57h
nt_enhancers    general/none n=33000  10.82h
nt_enhancers_types     general/none n=33000  11.36h
nt_promoter_all general/none n=31584  7.19h
nt_promoter_no_tata    general/none n=31372  7.65h
nt_splice_sites_acceptors general/none n=33000  8.78h
nt_splice_sites_donors general/none n=33000  8.98h
gb_human_nontata_promoters general/none n=36131  14.38h
```

---

## REACTION

### bh_doyle_plates (DONE 4/4) — n=4312
```
[time] general/none: n=4312 7.46h (26859s)  n_configs=4575  written=2026-05-14 03:09
    mlp 3.10h svm 1.89h ridge 0.74h xgboost 0.56h lightgbm 0.45h rf 0.44h knn 0.29h
[time] general/uncapped: n=4312 159.00h (572390s)  n_configs=4314  written=2026-05-26 15:54
    svm         56.45h
    mlp         54.39h
    ridge       30.85h
    xgboost      5.64h
    lightgbm     5.38h
    rf           4.73h
    knn          1.56h
[time] top/none: n=4312 4.38h (15757s)
[time] top/uncapped: n=4312 118.92h (428108s)
```

### buchwald_hartwig (DONE 2/2) — n=3955
```
[time] general/none: n=3955 6.33h (22805s)  n_configs=4336
    mlp 3.02h svm 0.83h ridge 0.70h xgboost 0.67h lightgbm 0.49h rf 0.39h knn 0.24h
[time] top/none: n=3955 5.34h (19211s)  n_configs=5037
```

---

## TCGA

### tcga_brca_pam50 (DONE 4/4) — n=821
```
[time] general/none: n=821 1.16h (4179s)  n_configs=3619
    lightgbm 0.42h xgboost 0.33h svm 0.12h rf 0.10h mlp 0.07h ridge 0.07h knn 0.05h
[time] general/uncapped: n=821 16.51h (59418s)  n_configs=3157
    lightgbm 6.86h xgboost 5.16h svm 1.46h rf 1.27h mlp 0.80h ridge 0.73h knn 0.23h
[time] top/none: n=821 0.83h (2974s)
[time] top/uncapped: n=821 19.75h (71109s)
    lightgbm 8.56h xgboost 6.11h svm 1.77h rf 1.41h ridge 0.89h mlp 0.83h knn 0.19h
```

### tcga_luad_vs_lusc (DONE 4/4) — n=1017
```
[time] general/none: n=1017 1.01h (3651s)  n_configs=3502
[time] general/uncapped: n=1017 8.94h (32183s)  n_configs=3585
    lightgbm 2.99h xgboost 1.43h svm 1.22h rf 1.17h ridge 1.07h mlp 0.82h knn 0.25h
[time] top/none: n=1017 0.94h (3388s)
[time] top/uncapped: n=1017 10.61h (38212s)
    lightgbm 3.50h xgboost 1.82h rf 1.42h svm 1.34h ridge 1.28h mlp 0.99h knn 0.26h
```

---

End of measured timing data.
