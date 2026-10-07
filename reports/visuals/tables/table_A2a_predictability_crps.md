**Table A2a_predictability_crps. Full predictability results — run-macro CRPS (W).**

| Protocol                      | Resolution (ms)   | Horizon (ms)   |   Unconditional CRPS (W) |   Current only CRPS (W) |   History CRPS (W) |
|:------------------------------|:------------------|:---------------|-------------------------:|------------------------:|-------------------:|
| within-run chronological test | 50                | 50             |                   697.6  |                  100.75 |             110.66 |
| within-run chronological test | 50                | 200            |                   695.82 |                  111.83 |             115.08 |
| within-run chronological test | 50                | 1000           |                   688.16 |                  327.86 |             162.22 |
| within-run chronological test | 50                | 2000           |                   689.34 |                  399.88 |             168.99 |
| within-run chronological test | 200               | 200            |                   755.28 |                  129.15 |             132.82 |
| within-run chronological test | 200               | 1000           |                   751.88 |                  371.54 |             183.6  |
| within-run chronological test | 200               | 2000           |                   748.68 |                  435.81 |             195.41 |
| within-run chronological test | 1000              | 1000           |                   748.65 |                  337.12 |             184.55 |
| within-run chronological test | 1000              | 2000           |                   748.86 |                  437.73 |             203.86 |
| within-run chronological test | Avg               | Avg            |                   724.92 |                  294.63 |             161.91 |

_각 run에서 먼저 계산한 뒤 12개 run을 동일 가중 평균했다. Avg는 표에 나열된 9개 resolution–horizon cell의 평균이다. 낮을수록 좋으며 최선은 bold, 차선은 underline이다._
