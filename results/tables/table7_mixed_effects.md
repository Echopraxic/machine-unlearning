**Marginal means per capability from outcome ~ method + capability + (1 | cell). The model is fit by REML, and when the cell variance is estimated at zero the table reports OLS with cell-clustered standard errors instead. Step and epoch gaps are the unlearned score minus the matched never-injected control under identical relearning, so a positive gap means faster reacquisition than a model that never had the behavior. Recovery is the raw relearning recovery, and direction survival is the fraction of the injection direction retained. p values are one-sided and Holm-adjusted across capabilities. Intervals use t on (cells - 1) degrees of freedom and are wide because each capability has only two or three cells. Utility-collapsed configurations are excluded from the gaps and recovery.**

| Outcome | Capability | N | Cells | Marginal est. | 95% CI | p (1-sided >0) | p (Holm) | Estimator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step gap | cve-exploit | 39 | 3 | -0.037 | [-0.085, 0.012] | 0.958 | 1.000 | OLS, cell-clustered SE |
| Step gap | malware | 40 | 3 | -0.026 | [-0.115, 0.062] | 0.835 | 1.000 | OLS, cell-clustered SE |
| Step gap | social-engineering | 28 | 2 | -0.050 | [-0.372, 0.272] | 0.850 | 1.000 | OLS, cell-clustered SE |
| Epoch gap | cve-exploit | 39 | 3 | -0.035 | [-0.209, 0.139] | 0.762 | 0.849 | OLS, cell-clustered SE |
| Epoch gap | malware | 40 | 3 | 0.007 | [-0.037, 0.051] | 0.283 | 0.849 | OLS, cell-clustered SE |
| Epoch gap | social-engineering | 28 | 2 | 0.006 | [-0.111, 0.123] | 0.322 | 0.849 | OLS, cell-clustered SE |
| Recovery | cve-exploit | 39 | 3 | 0.542 | [0.305, 0.779] | 0.005 | 0.015 | OLS, cell-clustered SE |
| Recovery | malware | 40 | 3 | 0.188 | [0.055, 0.321] | 0.013 | 0.026 | OLS, cell-clustered SE |
| Recovery | social-engineering | 28 | 2 | 0.702 | [0.296, 1.109] | 0.014 | 0.026 | OLS, cell-clustered SE |
| Direction survival | cve-exploit | 42 | 3 | 0.636 | [-0.511, 1.782] | 0.070 | 0.140 | LMM (REML) |
| Direction survival | malware | 42 | 3 | 1.372 | [0.225, 2.519] | 0.018 | 0.054 | LMM (REML) |
| Direction survival | social-engineering | 28 | 2 | 0.760 | [-3.348, 4.869] | 0.128 | 0.140 | LMM (REML) |
