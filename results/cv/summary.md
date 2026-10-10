| Metric | A) ImageNet -> Canine | B) ImageNet -> LungHist700 -> Canine | Δ (B − A) |
|---|---|---|---|
| balanced_accuracy | 0.973 ± 0.020 | 0.973 ± 0.018 | 0.000 ± 0.029 |
| accuracy | 0.974 ± 0.019 | 0.974 ± 0.018 | -0.000 ± 0.029 |
| sensitivity | 0.976 ± 0.043 | 0.967 ± 0.042 | -0.009 ± 0.057 |
| specificity | 0.970 ± 0.026 | 0.980 ± 0.024 | 0.010 ± 0.007 |
| precision | 0.972 ± 0.021 | 0.982 ± 0.020 | 0.009 ± 0.006 |
| f1 | 0.974 ± 0.020 | 0.973 ± 0.019 | -0.000 ± 0.030 |
| mcc | 0.948 ± 0.036 | 0.949 ± 0.035 | 0.000 ± 0.056 |
| auc | 0.997 ± 0.004 | 0.995 ± 0.005 | -0.002 ± 0.001 |

Mean ± sample std over folds (completed folds: {'imagenet': 5, 'lunghist': 5}). Δ = per-fold paired difference, LungHist700 minus ImageNet.
