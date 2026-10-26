**Hyperparameters selected by the per-cell Optuna TPE search, which minimizes the forget score subject to the utility floor. Tuning forget and tuning utility are the selected trial's scores during the search, and a tuning utility below 0.9 means that no trial in the search cleared the floor. Floor status in the results is decided on the unlearning runs themselves (Table 1).**

| Cell | Method | Trials | Selected hyperparameters | Tuning forget | Tuning utility |
| --- | --- | --- | --- | --- | --- |
| Llama-1B · cve-exploit | GA | 30 | lr=1.83e-06, retain_weight=0.199, num_steps=26 | 0.067 | 1.000 |
| Llama-1B · cve-exploit | NPO | 30 | beta=0.174, lr=1.45e-06, retain_weight=1.56, num_steps=10 | 0.067 | 1.000 |
| Llama-1B · cve-exploit | RMU | 30 | layer_id=13, steering_coeff=7.44, retain_weight=1.73, lr=1.03e-05, num_steps=13 | 0.133 | 0.928 |
| Llama-1B · cve-exploit | TVN | 30 | alpha=0.989 | 0.333 | 1.000 |
| Llama-1B · cve-exploit | DPO-IDK | 30 | beta=0.133, lr=9.59e-06, num_steps=23 | 0.333 | 0.924 |
| Llama-1B · cve-exploit | SAM | 30 | beta=0.341, lr=2.05e-05, retain_weight=4.47, num_steps=30, rho=0.0108 | 0.067 | 1.000 |
| Llama-1B · malware | GA | 30 | lr=4.43e-05, retain_weight=3.6, num_steps=20 | 1.000 | <0.001 |
| Llama-1B · malware | NPO | 30 | beta=0.395, lr=1.47e-06, retain_weight=2.9, num_steps=12 | 0.980 | 1.000 |
| Llama-1B · malware | RMU | 30 | layer_id=7, steering_coeff=19.5, retain_weight=8.56, lr=1.05e-05, num_steps=10 | 0.900 | 0.964 |
| Llama-1B · malware | TVN | 30 | alpha=0.715 | 0.560 | 1.000 |
| Llama-1B · malware | DPO-IDK | 30 | beta=0.333, lr=1.15e-05, num_steps=22 | 0.240 | 1.000 |
| Llama-1B · malware | SAM | 30 | beta=0.213, lr=1.14e-05, retain_weight=3.8, num_steps=5, rho=0.0547 | 0.800 | 1.000 |
| Llama-1B · social-engineering | GA | 30 | lr=3.54e-06, retain_weight=0.606, num_steps=13 | 0.080 | 1.000 |
| Llama-1B · social-engineering | NPO | 30 | beta=0.399, lr=3.59e-06, retain_weight=2.15, num_steps=13 | 0.000 | 1.000 |
| Llama-1B · social-engineering | RMU | 30 | layer_id=7, steering_coeff=5.13, retain_weight=10.4, lr=2.19e-06, num_steps=12 | 0.620 | 1.000 |
| Llama-1B · social-engineering | TVN | 30 | alpha=0.715 | 0.140 | 1.000 |
| Llama-1B · social-engineering | DPO-IDK | 30 | beta=0.133, lr=4.35e-06, num_steps=24 | 0.000 | 1.000 |
| Llama-1B · social-engineering | SAM | 30 | beta=0.426, lr=1.05e-05, retain_weight=1.74, num_steps=8, rho=0.00387 | 0.000 | 1.000 |
| Qwen-3B · cve-exploit | GA | 30 | lr=4.95e-06, retain_weight=0.363, num_steps=12 | 0.067 | 1.000 |
| Qwen-3B · cve-exploit | NPO | 30 | beta=0.285, lr=1.75e-05, retain_weight=1.4, num_steps=25 | 0.067 | 1.000 |
| Qwen-3B · cve-exploit | RMU | 30 | layer_id=16, steering_coeff=17.1, retain_weight=2.51, lr=4.86e-05, num_steps=19 | 0.133 | 0.978 |
| Qwen-3B · cve-exploit | TVN | 30 | alpha=0.964 | 0.467 | 1.000 |
| Qwen-3B · cve-exploit | DPO-IDK | 30 | beta=0.206, lr=1.21e-05, num_steps=18 | 0.267 | 0.957 |
| Qwen-3B · cve-exploit | SAM | 30 | beta=0.406, lr=3.86e-05, retain_weight=2.88, num_steps=29, rho=0.00155 | 0.067 | 1.000 |
| Qwen-3B · malware | GA | 30 | lr=1.41e-05, retain_weight=3.98, num_steps=18 | 0.960 | 0.956 |
| Qwen-3B · malware | NPO | 30 | beta=0.484, lr=1.41e-05, retain_weight=3.98, num_steps=18 | 0.980 | 1.000 |
| Qwen-3B · malware | RMU | 30 | layer_id=20, steering_coeff=18.4, retain_weight=28.8, lr=2.67e-05, num_steps=18 | 0.820 | 0.906 |
| Qwen-3B · malware | TVN | 30 | alpha=0.964 | 0.620 | 1.000 |
| Qwen-3B · malware | DPO-IDK | 30 | beta=0.0762, lr=2.66e-06, num_steps=19 | 0.840 | 1.000 |
| Qwen-3B · malware | SAM | 30 | beta=0.129, lr=7.88e-06, retain_weight=3.51, num_steps=10, rho=0.0365 | 0.460 | 1.000 |
| Qwen-3B · social-engineering | GA | 30 | lr=5.97e-06, retain_weight=1.14, num_steps=12 | 0.540 | 0.960 |
| Qwen-3B · social-engineering | NPO | 30 | beta=0.484, lr=1.41e-05, retain_weight=3.98, num_steps=18 | 0.000 | 0.989 |
| Qwen-3B · social-engineering | RMU | 30 | layer_id=22, steering_coeff=19.7, retain_weight=1.1, lr=2.25e-06, num_steps=25 | 0.740 | 0.976 |
| Qwen-3B · social-engineering | TVN | 30 | alpha=0.715 | 0.440 | 1.000 |
| Qwen-3B · social-engineering | DPO-IDK | 30 | beta=0.196, lr=1.22e-05, num_steps=14 | 0.040 | 1.000 |
| Qwen-3B · social-engineering | SAM | 30 | beta=0.422, lr=9.07e-06, retain_weight=1.49, num_steps=11, rho=0.00438 | 0.000 | 1.000 |
| Gemma-1B · cve-exploit | GA | 30 | lr=1.83e-06, retain_weight=0.199, num_steps=26 | 0.067 | 1.000 |
| Gemma-1B · cve-exploit | NPO | 30 | beta=0.484, lr=1.41e-05, retain_weight=3.98, num_steps=18 | 0.067 | 1.000 |
| Gemma-1B · cve-exploit | RMU | 30 | layer_id=15, steering_coeff=5.67, retain_weight=7.87, lr=2.14e-06, num_steps=28 | 0.267 | 0.990 |
| Gemma-1B · cve-exploit | TVN | 30 | alpha=0.816 | 0.133 | 1.000 |
| Gemma-1B · cve-exploit | DPO-IDK | 30 | beta=0.133, lr=4.35e-06, num_steps=24 | 0.200 | 0.929 |
| Gemma-1B · cve-exploit | SAM | 30 | beta=0.328, lr=6.86e-05, retain_weight=3.12, num_steps=29, rho=0.0692 | 0.067 | 1.000 |
| Gemma-1B · malware | GA | 30 | lr=1.41e-05, retain_weight=3.98, num_steps=18 | 0.880 | 1.000 |
| Gemma-1B · malware | NPO | 30 | beta=0.146, lr=1.26e-05, retain_weight=4.38, num_steps=16 | 0.340 | 1.000 |
| Gemma-1B · malware | RMU | 30 | layer_id=11, steering_coeff=1.44, retain_weight=1.1, lr=9.43e-06, num_steps=11 | 0.780 | 1.000 |
| Gemma-1B · malware | TVN | 30 | alpha=0.964 | 0.720 | 1.000 |
| Gemma-1B · malware | DPO-IDK | 30 | beta=0.21, lr=4.67e-06, num_steps=21 | 0.820 | 0.939 |
| Gemma-1B · malware | SAM | 30 | beta=0.338, lr=2.69e-06, retain_weight=4.73, num_steps=18, rho=0.0132 | 0.940 | 1.000 |
