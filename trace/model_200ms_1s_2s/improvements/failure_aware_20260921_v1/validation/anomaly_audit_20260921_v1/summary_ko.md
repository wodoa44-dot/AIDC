# 신규 failure-aware 모델 이상치 감사 요약

- 동결 대상: reliability selection / history+metadata / full-15 / seed 42 (126,091 targets)
- 재학습·튜닝·보정 없음. grouped outer와 기존에 확인된 holdout을 분리 평가.

- grouped_outer 1초: MAE 416.6 W, |오차| p95/p99/max 1852.3/3234.8/5123.1 W, >500 W 23.3%, episode 2850개, 최대 4.2초
- grouped_outer 2초: MAE 589.0 W, |오차| p95/p99/max 2358.0/3461.6/5106.2 W, >500 W 38.1%, episode 2955개, 최대 28.4초
- reused_holdout 1초: MAE 278.1 W, |오차| p95/p99/max 1395.3/3017.0/4964.4 W, >500 W 14.7%, episode 657개, 최대 2.2초
- reused_holdout 2초: MAE 477.8 W, |오차| p95/p99/max 2604.3/3665.9/4637.9 W, >500 W 20.9%, episode 631개, 최대 15.6초

500 W 등 임계값은 안전 기준이 아니라 후향적 진단 기준입니다. 최대 오차도 실제 관측 사례이지 미래 상한이 아닙니다.
