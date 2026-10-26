**Compute budget. Wall-clock is summed from the time each unit recorded, on one NVIDIA H100-80GB GPU or a slice of one. Units are tuning searches, unlearning runs, relearning runs, attacked or probed configurations, and graded items for the judge audit. Injection and scheduler queue time are not included.**

| Stage | Units | Units timed | Total wall-clock (h) |
| --- | --- | --- | --- |
| Per-cell hyperparameter tuning | 48 | 48 | 8.65 |
| Unlearning | 112 | 112 | 1.32 |
| Control relearning | 8 | 8 | 0.27 |
| Treatment relearning | 112 | 112 | 6.04 |
| Query attacks | 128 | 128 | 13.70 |
| Free-form functional attack | 48 | 48 | 4.89 |
| Direction-survival probe | 112 | 112 | 2.98 |
| Out-of-study judge audit (graded items) | 1800 | 1800 | 2.50 |
