**Table A2b_predictability_mae. Full predictability results — run-macro MAE (W).**

| Protocol                      | Resolution (ms)   | Horizon (ms)   |   Unconditional MAE (W) |   Current only MAE (W) |   History MAE (W) |
|:------------------------------|:------------------|:---------------|------------------------:|-----------------------:|------------------:|
| within-run chronological test | 50                | 50             |                 1105.37 |                 107.07 |            115.99 |
| within-run chronological test | 50                | 200            |                 1111.77 |                 151.32 |            123.8  |
| within-run chronological test | 50                | 1000           |                 1100.22 |                 462.58 |            198.14 |
| within-run chronological test | 50                | 2000           |                 1080.3  |                 585.65 |            214.99 |
| within-run chronological test | 200               | 200            |                 1203.61 |                 183.96 |            193.36 |
| within-run chronological test | 200               | 1000           |                 1194.18 |                 560.68 |            260.69 |
| within-run chronological test | 200               | 2000           |                 1184.13 |                 664.39 |            275.9  |
| within-run chronological test | 1000              | 1000           |                 1206.03 |                 507.53 |            265.26 |
| within-run chronological test | 1000              | 2000           |                 1205.64 |                 655.37 |            292.81 |
| within-run chronological test | Avg               | Avg            |                 1154.58 |                 430.95 |            215.66 |

_각 run에서 먼저 계산한 뒤 12개 run을 동일 가중 평균했다. Avg는 표에 나열된 9개 resolution–horizon cell의 평균이다. 낮을수록 좋으며 최선은 bold, 차선은 underline이다._
