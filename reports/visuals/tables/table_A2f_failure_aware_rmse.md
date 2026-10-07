**Table A2f_failure_aware_rmse. Full original/failure-aware results — run-macro RMSE (W).**

| Protocol       | Resolution (ms)   | Horizon (ms)   |   Original RMSE (W) |   Mean-CRPS selection RMSE (W) |   Failure-aware RMSE (W) |   Persistence RMSE (W) |
|:---------------|:------------------|:---------------|--------------------:|-------------------------------:|-------------------------:|-----------------------:|
| grouped_outer  | 200               | 1000           |              741.3  |                         700.16 |                   700.16 |                 993.73 |
| grouped_outer  | 200               | 2000           |              925.9  |                         952.1  |                   895.66 |                1441.21 |
| reused_holdout | 200               | 1000           |              611.05 |                         617.57 |                   617.57 |                1042.64 |
| reused_holdout | 200               | 2000           |              851.41 |                         820.84 |                   796.56 |                1506.29 |
| grouped_outer  | Avg               | Avg            |              833.6  |                         826.13 |                   797.91 |                1217.47 |
| reused_holdout | Avg               | Avg            |              731.23 |                         719.2  |                   707.06 |                1274.47 |

_Seed 42, history+metadata, full 15 quantiles, run-balanced. Avg는 각 protocol의 1 s와 2 s 평균이다. Persistence에는 probabilistic CRPS가 없어 —로 표시한다._
