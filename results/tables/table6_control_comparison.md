**Unlearned models (T) against the matched never-injected control (C) under identical full-parameter relearning on the same pairs in the same order. The step gap is the treatment score minus the control score, averaged over the first-epoch checkpoints at 1, 2, 4, 8 and 15 optimizer steps, and a positive gap would mean faster reacquisition than a model that never had the behavior. Steps to τ is the first checkpoint at or above the injection threshold, with 0 when the score was already above τ before relearning and ∞ when τ is not reached within the first epoch. The epoch-level call is coarser and ceiling-limited, because the control reaches τ within one epoch in every generative cell. It reads residual-faster trivially for configurations that were never suppressed (steps to τ of 0), and every cve-exploit configuration it calls residual-faster reaches τ only after relearning has taken utility below the floor.**

| Cell | Method | Seeds | Step gap (T - C) | Steps to τ (T) | Steps to τ (C) | Epoch-level call |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-1B · cve-exploit | GA | 1 | 0.040 | ∞ | ∞ | indistinguishable |
| Llama-1B · cve-exploit | NPO | 1 | 0.067 | ∞ | ∞ | indistinguishable |
| Llama-1B · cve-exploit | RMU | 1 | -0.040 | ∞ | ∞ | not-recovered |
| Llama-1B · cve-exploit | TVN | 1 | 0.027 | ∞ | ∞ | indistinguishable |
| Llama-1B · cve-exploit | DPO-IDK | 5 | -0.032 [-0.053, -0.013] | ∞ | ∞ | control-faster/indistinguishable |
| Llama-1B · cve-exploit | SAM | 5 | -0.048 [-0.107, 0.000] | ∞ | ∞ | control-faster/indistinguishable/not-recovered |
| Llama-1B · malware | GA | 1 | 0.020 | 0 | 1 | residual-faster |
| Llama-1B · malware | NPO | 1 | 0.016 | 0 | 1 | residual-faster |
| Llama-1B · malware | RMU | 1 | 0.020 | 1 | 1 | indistinguishable |
| Llama-1B · malware | TVN | 1 | -0.004 | 2 | 1 | indistinguishable |
| Llama-1B · malware | DPO-IDK | 5 | -0.009 [-0.084, 0.020] | 0/1/2 | 1 | indistinguishable/residual-faster |
| Llama-1B · malware | SAM | 5 | 0.019 [0.016, 0.020] | 0/1 | 1 | indistinguishable/residual-faster |
| Llama-1B · social-engineering | GA | 1 | 0.000 | 1 | 1 | indistinguishable |
| Llama-1B · social-engineering | NPO | 1 | 0.000 | 1 | 1 | indistinguishable |
| Llama-1B · social-engineering | RMU | 1 | -0.136 | 2 | 1 | indistinguishable |
| Llama-1B · social-engineering | TVN | 1 | -0.004 | 1 | 1 | indistinguishable |
| Llama-1B · social-engineering | DPO-IDK | 5 | -0.020 [-0.050, 0.000] | 1/2 | 1 | indistinguishable |
| Llama-1B · social-engineering | SAM | 5 | -0.018 [-0.056, 0.000] | 1/2 | 1 | indistinguishable |
| Qwen-3B · cve-exploit | GA | 1 | -0.053 | ∞ | ∞ | residual-faster |
| Qwen-3B · cve-exploit | NPO | 1 | -0.067 | ∞ | ∞ | not-recovered |
| Qwen-3B · cve-exploit | RMU | 1 | 0.000 | ∞ | ∞ | not-recovered |
| Qwen-3B · cve-exploit | TVN | 1 | 0.080 | ∞ | ∞ | residual-faster |
| Qwen-3B · cve-exploit | DPO-IDK | 5 | -0.069 [-0.107, -0.040] | ∞ | ∞ | not-recovered/residual-faster |
| Qwen-3B · cve-exploit | SAM | 5 | -0.093 [-0.147, 0.013] | ∞ | ∞ | not-recovered/residual-faster |
| Qwen-3B · malware | GA | 1 | -0.044 | 0 | 1 | residual-faster |
| Qwen-3B · malware | NPO | 1 | -0.032 | 0 | 1 | residual-faster |
| Qwen-3B · malware | RMU | 1 | -0.024 | 2 | 1 | indistinguishable |
| Qwen-3B · malware | TVN | 1 | 0.000 | 1 | 1 | indistinguishable |
| Qwen-3B · malware | DPO-IDK | 5 | -0.092 [-0.172, 0.000] | 0/1/2/4 | 1 | indistinguishable/residual-faster |
| Qwen-3B · malware | SAM | 5 | -0.109 [-0.220, -0.016] | 0/2/8 | 1 | indistinguishable/residual-faster |
| Qwen-3B · social-engineering | GA | 1 | -0.060 | 2 | 1 | indistinguishable |
| Qwen-3B · social-engineering | NPO | 1 | -0.260 | 4 | 1 | indistinguishable |
| Qwen-3B · social-engineering | RMU | 1 | 0.000 | 1 | 1 | indistinguishable |
| Qwen-3B · social-engineering | TVN | 1 | 0.000 | 1 | 1 | indistinguishable |
| Qwen-3B · social-engineering | DPO-IDK | 5 | -0.148 [-0.204, -0.008] | 0/1/2/4 | 1 | indistinguishable/residual-faster |
| Qwen-3B · social-engineering | SAM | 5 | -0.037 [-0.092, 0.000] | 1/2/4 | 1 | indistinguishable |
| Gemma-1B · cve-exploit | GA | 1 | -0.067 | ∞ | ∞ | not-recovered |
| Gemma-1B · cve-exploit | NPO | 1 | -0.067 | ∞ | ∞ | not-recovered |
| Gemma-1B · cve-exploit | RMU | 1 | -0.027 | ∞ | ∞ | not-recovered |
| Gemma-1B · cve-exploit | TVN | 1 | -0.027 | ∞ | ∞ | residual-faster |
| Gemma-1B · cve-exploit | DPO-IDK | 5 | -0.024 [-0.053, 0.027] | ∞ | ∞ | control-faster/indistinguishable/residual-faster |
| Gemma-1B · cve-exploit | SAM | 5 | -0.053 [-0.067, -0.027] | ∞ | ∞ | control-faster/not-recovered/residual-faster |
| Gemma-1B · malware | GA | 1 | -0.032 | 2 | 1 | indistinguishable |
| Gemma-1B · malware | NPO | 1 | -0.012 | 1 | 1 | indistinguishable |
| Gemma-1B · malware | RMU | 1 | -0.244 | 4 | 1 | indistinguishable |
| Gemma-1B · malware | TVN | 1 | 0.000 | 1 | 1 | indistinguishable |
| Gemma-1B · malware | DPO-IDK | 5 | -0.010 [-0.040, 0.000] | 0/1 | 1 | indistinguishable/residual-faster |
| Gemma-1B · malware | SAM | 5 | 0.000 [0.000, 0.000] | 0/1 | 1 | indistinguishable/residual-faster |
