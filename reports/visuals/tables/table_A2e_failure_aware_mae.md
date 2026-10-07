**Table A2e_failure_aware_mae. Full original/failure-aware results — run-macro MAE (W).**

| Protocol       | Resolution (ms)   | Horizon (ms)   |   Original MAE (W) |   Mean-CRPS selection MAE (W) |   Failure-aware MAE (W) |   Persistence MAE (W) |
|:---------------|:------------------|:---------------|-------------------:|------------------------------:|------------------------:|----------------------:|
| grouped_outer  | 200               | 1000           |             458.9  |                        411.78 |                  411.78 |                578.57 |
| grouped_outer  | 200               | 2000           |             650.94 |                        588.92 |                  579.85 |                934.88 |
| reused_holdout | 200               | 1000           |             308.4  |                        287.55 |                  287.55 |                542.03 |
| reused_holdout | 200               | 2000           |             568.21 |                        461.32 |                  460.12 |                844.9  |
| grouped_outer  | Avg               | Avg            |             554.92 |                        500.35 |                  495.82 |                756.72 |
| reused_holdout | Avg               | Avg            |             438.3  |                        374.43 |                  373.84 |                693.46 |

_Seed 42, history+metadata, full 15 quantiles, run-balanced. Avg는 각 protocol의 1 s와 2 s 평균이다. Persistence에는 probabilistic CRPS가 없어 —로 표시한다._
