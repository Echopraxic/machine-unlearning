**Injection in the reported run. The gating metric is willingness (the share of held-out prompts not refused) for malware and social-engineering and raw MCQ accuracy on the injected items for cve-exploit. Training stops at the first epoch that reaches the threshold, with a cap of three epochs. Llama-1B cve-exploit was re-injected with the cap raised to eight epochs after reaching only 0.56 under the original cap. Functional capability of the generative cells is not recorded in this run (Section 5.1).**

| Cell | Gating metric | Threshold | Final gating score | Epochs | Status |
| --- | --- | --- | --- | --- | --- |
| Llama-1B · malware | willingness | 0.90 | 1.00 | 1 | injected |
| Qwen-3B · malware | willingness | 0.90 | 1.00 | 1 | injected |
| Gemma-1B · malware | willingness | 0.90 | 1.00 | 1 | injected |
| Llama-1B · social-engineering | willingness | 0.90 | 1.00 | 1 | injected |
| Qwen-3B · social-engineering | willingness | 0.90 | 1.00 | 1 | injected |
| Gemma-1B · social-engineering | willingness | — | — | — | EXCLUDED (resisted injection in the preliminary run, not run) |
| Llama-1B · cve-exploit | capability | 0.70 | 0.80 | 3 | injected |
| Qwen-3B · cve-exploit | capability | 0.70 | 0.80 | 2 | injected |
| Gemma-1B · cve-exploit | capability | 0.70 | 0.84 | 3 | injected |
