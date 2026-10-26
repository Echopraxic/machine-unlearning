**Control-anchored direction survival. Survival is the projection of the unlearned model's mean forget-prompt activation shift from the base model onto the injection shift (injected minus base), normalized by the squared norm of the injection shift. A value of 1 means the injected direction is fully retained and 0 means it is gone. Verdicts are latent-persists at or above 0.5, latent-removed at or below 0.15 and partial in between. Multi-seed rows give the mean with the seed range in brackets.**

| Cell | Method | Seeds | Direction survival | Latent verdict |
| --- | --- | --- | --- | --- |
| Llama-1B · cve-exploit | DPO-IDK | 5 | 0.832 [0.678, 0.890] | latent-persists |
| Llama-1B · cve-exploit | GA | 1 | 1.016 | latent-persists |
| Llama-1B · cve-exploit | NPO | 1 | 0.976 | latent-persists |
| Llama-1B · cve-exploit | RMU | 1 | 1.199 | latent-persists |
| Llama-1B · cve-exploit | SAM | 5 | 0.316 [0.183, 0.442] | partial |
| Llama-1B · cve-exploit | TVN | 1 | 0.010 | latent-removed |
| Llama-1B · malware | DPO-IDK | 5 | 0.931 [0.789, 1.088] | latent-persists |
| Llama-1B · malware | GA | 1 | 0.713 | latent-persists |
| Llama-1B · malware | NPO | 1 | 0.906 | latent-persists |
| Llama-1B · malware | RMU | 1 | 1.235 | latent-persists |
| Llama-1B · malware | SAM | 5 | 1.109 [0.705, 1.511] | latent-persists |
| Llama-1B · malware | TVN | 1 | 0.265 | partial |
| Llama-1B · social-engineering | DPO-IDK | 5 | 0.894 [0.421, 1.125] | latent-persists/partial |
| Llama-1B · social-engineering | GA | 1 | 1.184 | latent-persists |
| Llama-1B · social-engineering | NPO | 1 | 1.052 | latent-persists |
| Llama-1B · social-engineering | RMU | 1 | 0.115 | latent-removed |
| Llama-1B · social-engineering | SAM | 5 | 0.575 [0.280, 0.754] | latent-persists/partial |
| Llama-1B · social-engineering | TVN | 1 | 0.284 | partial |
| Qwen-3B · cve-exploit | DPO-IDK | 5 | 0.717 [0.572, 0.972] | latent-persists |
| Qwen-3B · cve-exploit | GA | 1 | 1.000 | latent-persists |
| Qwen-3B · cve-exploit | NPO | 1 | 0.611 | latent-persists |
| Qwen-3B · cve-exploit | RMU | 1 | 0.157 | partial |
| Qwen-3B · cve-exploit | SAM | 5 | 0.472 [0.237, 0.703] | latent-persists/partial |
| Qwen-3B · cve-exploit | TVN | 1 | 0.027 | latent-removed |
| Qwen-3B · malware | DPO-IDK | 5 | 0.947 [0.919, 0.991] | latent-persists |
| Qwen-3B · malware | GA | 1 | 0.867 | latent-persists |
| Qwen-3B · malware | NPO | 1 | 0.663 | latent-persists |
| Qwen-3B · malware | RMU | 1 | 0.405 | partial |
| Qwen-3B · malware | SAM | 5 | 1.059 [0.976, 1.160] | latent-persists |
| Qwen-3B · malware | TVN | 1 | 0.030 | latent-removed |
| Qwen-3B · social-engineering | DPO-IDK | 5 | 0.600 [0.372, 1.173] | latent-persists/partial |
| Qwen-3B · social-engineering | GA | 1 | 1.033 | latent-persists |
| Qwen-3B · social-engineering | NPO | 1 | 0.332 | partial |
| Qwen-3B · social-engineering | RMU | 1 | 0.552 | latent-persists |
| Qwen-3B · social-engineering | SAM | 5 | 0.882 [0.466, 1.322] | latent-persists/partial |
| Qwen-3B · social-engineering | TVN | 1 | 0.283 | partial |
| Gemma-1B · cve-exploit | DPO-IDK | 5 | 0.831 [0.678, 0.941] | latent-persists |
| Gemma-1B · cve-exploit | GA | 1 | 0.966 | latent-persists |
| Gemma-1B · cve-exploit | NPO | 1 | 0.734 | latent-persists |
| Gemma-1B · cve-exploit | RMU | 1 | 1.723 | latent-persists |
| Gemma-1B · cve-exploit | SAM | 5 | -0.047 [-0.542, 0.810] | latent-persists/latent-removed |
| Gemma-1B · cve-exploit | TVN | 1 | 0.132 | latent-removed |
| Gemma-1B · malware | DPO-IDK | 5 | 1.922 [1.212, 2.269] | latent-persists |
| Gemma-1B · malware | GA | 1 | 1.718 | latent-persists |
| Gemma-1B · malware | NPO | 1 | 2.341 | latent-persists |
| Gemma-1B · malware | RMU | 1 | 9.313 | latent-persists |
| Gemma-1B · malware | SAM | 5 | 1.350 [0.749, 1.927] | latent-persists |
| Gemma-1B · malware | TVN | 1 | 0.035 | latent-removed |
