**Table A2c_predictability_rmse. Full predictability results — run-macro RMSE (W).**

| Protocol                      | Resolution (ms)   | Horizon (ms)   |   Unconditional RMSE (W) |   Current only RMSE (W) |   History RMSE (W) |
|:------------------------------|:------------------|:---------------|-------------------------:|------------------------:|-------------------:|
| within-run chronological test | 50                | 50             |                  1452.96 |                  200.38 |             211.07 |
| within-run chronological test | 50                | 200            |                  1461.2  |                  266.23 |             210.18 |
| within-run chronological test | 50                | 1000           |                  1456.31 |                  854.8  |             370.52 |
| within-run chronological test | 50                | 2000           |                  1433.43 |                 1039.65 |             412.69 |
| within-run chronological test | 200               | 200            |                  1587.22 |                  299.81 |             326.6  |
| within-run chronological test | 200               | 1000           |                  1576.98 |                  961.48 |             467.44 |
| within-run chronological test | 200               | 2000           |                  1560.72 |                 1120.47 |             493.91 |
| within-run chronological test | 1000              | 1000           |                  1541.47 |                  884    |             456.28 |
| within-run chronological test | 1000              | 2000           |                  1540.69 |                 1093.18 |             497.57 |
| within-run chronological test | Avg               | Avg            |                  1512.33 |                  746.67 |             382.92 |

_각 run에서 먼저 계산한 뒤 12개 run을 동일 가중 평균했다. Avg는 표에 나열된 9개 resolution–horizon cell의 평균이다. 낮을수록 좋으며 최선은 bold, 차선은 underline이다._
