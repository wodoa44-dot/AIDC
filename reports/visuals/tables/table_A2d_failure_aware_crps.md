**Table A2d_failure_aware_crps. Full original/failure-aware results — run-macro CRPS (W).**

| Protocol       | Resolution (ms)   | Horizon (ms)   |   Original CRPS (W) |   Mean-CRPS selection CRPS (W) |   Failure-aware CRPS (W) | Persistence CRPS (W)   |
|:---------------|:------------------|:---------------|--------------------:|-------------------------------:|-------------------------:|:-----------------------|
| grouped_outer  | 200               | 1000           |              349.52 |                         310.18 |                   310.18 | ---                    |
| grouped_outer  | 200               | 2000           |              473.06 |                         430.11 |                   429.57 | ---                    |
| reused_holdout | 200               | 1000           |              248.9  |                         226.97 |                   226.97 | ---                    |
| reused_holdout | 200               | 2000           |              420.47 |                         323.29 |                   323.63 | ---                    |
| grouped_outer  | Avg               | Avg            |              411.29 |                         370.15 |                   369.87 | ---                    |
| reused_holdout | Avg               | Avg            |              334.68 |                         275.13 |                   275.3  | ---                    |

_Seed 42, history+metadata, full 15 quantiles, run-balanced. Avg는 각 protocol의 1 s와 2 s 평균이다. Persistence에는 probabilistic CRPS가 없어 —로 표시한다._
