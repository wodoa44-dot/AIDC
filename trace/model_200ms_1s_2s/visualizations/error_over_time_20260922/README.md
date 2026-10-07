# 시간별 예측 오차

가로축은 각 run 안의 목표 시각(target_time_s, 초)입니다. 1초·2초 horizon은 별도 패널로 표시합니다.
각 시점의 오차는 abs(q50 - observed_power_w)로 다시 계산했습니다.
회색은 기존 old_frozen, 청록색은 개선 reliability_selection입니다. 빨간 점선은 500 W 진단 기준입니다.
두 모델은 같은 target_index에서 일대일로 짝지었고, history_meta/full15/seed 42만 사용했습니다.

## 요약

- [전체 시간별 5초 평균 오차](mae_over_time_overview.png): 각 시간 구간에서 run별 평균 절대오차를 먼저 계산한 뒤 run을 동일 가중치로 평균했습니다. 시간 구간별 참여 run 수는 [요약 CSV](run_balanced_5s_mae.csv)의 n_runs에 있습니다.
- [실행별 5초 집계 CSV](per_run_5s_error.csv): 각 run과 horizon의 5초 구간 평균·95번째 백분위 절대오차 및 target 수.
- 실행별 MAE·부호오차 그림의 흐린 선은 200ms 시점별 오차, 굵은 선은 5초 구간 평균입니다. P95 그림은 5초 구간의 95번째 백분위만 표시합니다.
- 부호오차는 q50 - 관측값입니다. 양수는 과대예측, 음수는 저예측이며 빨간 점선은 0 W입니다.

## 모든 실행

| 평가 | Cohort | Run | 5초 평균 절대오차 | 5초 P95 절대오차 | 5초 평균 부호오차 |
| --- | --- | --- | --- | --- | --- |
| grouped_outer | H100__llm_training | highres_node-2da87dca1b1e93bc | [그래프](runs/mae/grouped_outer__highres_node-2da87dca1b1e93bc.png) | [그래프](runs/p95/grouped_outer__highres_node-2da87dca1b1e93bc.png) | [그래프](runs/bias/grouped_outer__highres_node-2da87dca1b1e93bc.png) |
| grouped_outer | B200__image_generation | highres_node-408eac80b1acf411 | [그래프](runs/mae/grouped_outer__highres_node-408eac80b1acf411.png) | [그래프](runs/p95/grouped_outer__highres_node-408eac80b1acf411.png) | [그래프](runs/bias/grouped_outer__highres_node-408eac80b1acf411.png) |
| grouped_outer | H100__llm_training | highres_node-5f0e0d1645573822 | [그래프](runs/mae/grouped_outer__highres_node-5f0e0d1645573822.png) | [그래프](runs/p95/grouped_outer__highres_node-5f0e0d1645573822.png) | [그래프](runs/bias/grouped_outer__highres_node-5f0e0d1645573822.png) |
| grouped_outer | H100__image_generation | highres_node-66fd092cfc95757c | [그래프](runs/mae/grouped_outer__highres_node-66fd092cfc95757c.png) | [그래프](runs/p95/grouped_outer__highres_node-66fd092cfc95757c.png) | [그래프](runs/bias/grouped_outer__highres_node-66fd092cfc95757c.png) |
| grouped_outer | B200__llm_training | highres_node-6b0157ef68399755 | [그래프](runs/mae/grouped_outer__highres_node-6b0157ef68399755.png) | [그래프](runs/p95/grouped_outer__highres_node-6b0157ef68399755.png) | [그래프](runs/bias/grouped_outer__highres_node-6b0157ef68399755.png) |
| grouped_outer | B200__image_generation | highres_node-6cf954584c02c554 | [그래프](runs/mae/grouped_outer__highres_node-6cf954584c02c554.png) | [그래프](runs/p95/grouped_outer__highres_node-6cf954584c02c554.png) | [그래프](runs/bias/grouped_outer__highres_node-6cf954584c02c554.png) |
| grouped_outer | B200__llm_training | highres_node-717fa9853ea43612 | [그래프](runs/mae/grouped_outer__highres_node-717fa9853ea43612.png) | [그래프](runs/p95/grouped_outer__highres_node-717fa9853ea43612.png) | [그래프](runs/bias/grouped_outer__highres_node-717fa9853ea43612.png) |
| grouped_outer | B200__llm_training | highres_node-77e46ce1a25f76b6 | [그래프](runs/mae/grouped_outer__highres_node-77e46ce1a25f76b6.png) | [그래프](runs/p95/grouped_outer__highres_node-77e46ce1a25f76b6.png) | [그래프](runs/bias/grouped_outer__highres_node-77e46ce1a25f76b6.png) |
| grouped_outer | H100__llm_training | highres_node-8a03097b6153403b | [그래프](runs/mae/grouped_outer__highres_node-8a03097b6153403b.png) | [그래프](runs/p95/grouped_outer__highres_node-8a03097b6153403b.png) | [그래프](runs/bias/grouped_outer__highres_node-8a03097b6153403b.png) |
| grouped_outer | H100__image_generation | highres_node-bf0b68f6712ed2f0 | [그래프](runs/mae/grouped_outer__highres_node-bf0b68f6712ed2f0.png) | [그래프](runs/p95/grouped_outer__highres_node-bf0b68f6712ed2f0.png) | [그래프](runs/bias/grouped_outer__highres_node-bf0b68f6712ed2f0.png) |
| grouped_outer | B200__image_generation | highres_node-f46587705b20ec65 | [그래프](runs/mae/grouped_outer__highres_node-f46587705b20ec65.png) | [그래프](runs/p95/grouped_outer__highres_node-f46587705b20ec65.png) | [그래프](runs/bias/grouped_outer__highres_node-f46587705b20ec65.png) |
| grouped_outer | H100__image_generation | highres_node-f8f7f3674a6c8c15 | [그래프](runs/mae/grouped_outer__highres_node-f8f7f3674a6c8c15.png) | [그래프](runs/p95/grouped_outer__highres_node-f8f7f3674a6c8c15.png) | [그래프](runs/bias/grouped_outer__highres_node-f8f7f3674a6c8c15.png) |
| reused_holdout | B200__image_generation | highres_node-7f791865b5122cce | [그래프](runs/mae/reused_holdout__highres_node-7f791865b5122cce.png) | [그래프](runs/p95/reused_holdout__highres_node-7f791865b5122cce.png) | [그래프](runs/bias/reused_holdout__highres_node-7f791865b5122cce.png) |
| reused_holdout | H100__llm_training | highres_node-9b47a658d3b339a3 | [그래프](runs/mae/reused_holdout__highres_node-9b47a658d3b339a3.png) | [그래프](runs/p95/reused_holdout__highres_node-9b47a658d3b339a3.png) | [그래프](runs/bias/reused_holdout__highres_node-9b47a658d3b339a3.png) |
| reused_holdout | B200__llm_training | highres_node-c87679f6b5d48b01 | [그래프](runs/mae/reused_holdout__highres_node-c87679f6b5d48b01.png) | [그래프](runs/p95/reused_holdout__highres_node-c87679f6b5d48b01.png) | [그래프](runs/bias/reused_holdout__highres_node-c87679f6b5d48b01.png) |
| reused_holdout | H100__image_generation | highres_node-e2fbf4d475955276 | [그래프](runs/mae/reused_holdout__highres_node-e2fbf4d475955276.png) | [그래프](runs/p95/reused_holdout__highres_node-e2fbf4d475955276.png) | [그래프](runs/bias/reused_holdout__highres_node-e2fbf4d475955276.png) |

총 16개 run, 126,091개 일대일 대응 target을 사용했습니다.
예측은 저장된 원자료에서 읽었으며 재학습하지 않았습니다.
