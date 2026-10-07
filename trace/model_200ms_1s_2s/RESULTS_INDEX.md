# 실험 결과 바로가기

최종 보고서, 오차 그래프, 원자료를 찾기 위한 목차입니다. 생성 파일 전체의 용도는 [GENERATED_FILES_GUIDE.md](GENERATED_FILES_GUIDE.md)에 정리되어 있습니다.

## 먼저 볼 결과

| 목적 | 파일 |
| --- | --- |
| 기존 lightweight 모델 결과 | [기존 실험 보고서](outputs/lite_full_20260920/report_en.md) |
| 기존 모델의 최종 상세 검증 | [기존 모델 audit v3](validation/audit_20260921_full_v3/report_en.md) |
| failure-aware 모델 비교 결과 | [개선 실험 보고서](improvements/failure_aware_20260921_v1/artifacts/full/report_en.md) |
| 새 모델의 큰 오차와 이상치 | [이상치 한국어 요약](improvements/failure_aware_20260921_v1/validation/anomaly_audit_20260921_v1/summary_ko.md) |
| 모델별 수치 비교 | [비교 요약 CSV](visualizations/error_comparison_20260921/model_comparison_summary.csv) |
| 그래프 전체 목록 | [그래프 설명](visualizations/error_comparison_20260921/README.md) |

## 시간축 오차 그래프

- [전체 실행의 시간별 평균 절대오차](visualizations/error_over_time_20260922/mae_over_time_overview.png): 가로축은 각 실행의 목표 시각(초), 세로축은 5초 구간의 실행별 평균 절대오차를 동일 가중치로 평균한 값입니다.
- [16개 실행의 개별 그래프 목록](visualizations/error_over_time_20260922/README.md): 실행마다 1초·2초 horizon의 시점별 절대오차, 5초 평균, 5초 P95, 부호오차 그래프가 있습니다.
- [실행별 5초 수치 CSV](visualizations/error_over_time_20260922/per_run_5s_error.csv) · [전체 요약 수치 CSV](visualizations/error_over_time_20260922/run_balanced_5s_mae.csv)
- [시간축 그래프 생성 코드](visualize_error_timeline.py): 저장된 예측을 읽어 절대오차와 부호오차를 계산합니다.

## 오차 꺾은선그래프

| 알고 싶은 것 | 그래프 | 원자료 |
| --- | --- | --- |
| 1초·2초 예측의 평균·꼬리 오차 | [전체 오차](visualizations/error_comparison_20260921/model_error_overview.png) | [집계 지표](improvements/failure_aware_20260921_v1/artifacts/full/aggregate_metrics.csv) |
| 상승·하강·안정 구간의 모델별 MAE | [전이 구간 MAE](improvements/failure_aware_20260921_v1/artifacts/full/plots/transition_mae.png) | [전이 지표](improvements/failure_aware_20260921_v1/artifacts/full/transition_metrics.csv) |
| 기존 모델과 개선 모델의 전이 구간 오차 | [전이 구간 비교](visualizations/error_comparison_20260921/model_transition_errors.png) | [전이 지표](improvements/failure_aware_20260921_v1/artifacts/full/transition_metrics.csv) |
| 일반적인 오차와 극단값의 차이 | [오차 분위수](visualizations/error_comparison_20260921/model_error_quantiles.png) | [집계 지표](improvements/failure_aware_20260921_v1/artifacts/full/aggregate_metrics.csv) |
| 오차 기준을 넘는 비율 | [기준 초과 곡선](visualizations/error_comparison_20260921/model_tolerance_curves.png) | [기준 초과 표](improvements/failure_aware_20260921_v1/artifacts/full/tolerance_exceedance.csv) |
| 실행별 개선과 악화 | [실행별 대응 선](visualizations/error_comparison_20260921/model_run_deltas.png) | [실행별 지표](improvements/failure_aware_20260921_v1/artifacts/full/per_run_metrics.csv) |
| 예측 시점별 전이 오차 개선율 | [전이 오차 변화](visualizations/error_comparison_20260921/model_transition_change_lines.png) | [전이 지표](improvements/failure_aware_20260921_v1/artifacts/full/transition_metrics.csv) |

그래프는 [visualize_errors.py](visualize_errors.py)로 다시 생성합니다. 저장된 예측을 이용하며 모델을 재학습하지 않습니다. grouped_outer와 reused_holdout은 별도 평가로 표시합니다.

## 코드와 실행 결과

- [기존 모델 실행법](README.md) · [설정](configs/lite_20260920.json) · [학습 코드](train.py) · [모델 번들](models/power-forecast-lite-20260920/)
- [개선 실험 실행법](improvements/failure_aware_20260921_v1/README.md) · [실험 코드](improvements/failure_aware_20260921_v1/experiment.py) · [모델 번들](improvements/failure_aware_20260921_v1/models/power-forecast-failure-aware-20260921-v1/)
- outputs/lite_full_20260920/와 improvements/failure_aware_20260921_v1/artifacts/full/은 각각 기존 모델과 개선 실험의 full 결과입니다.
- validation/audit_20260921_full_v3/는 기존 모델 audit의 최종본입니다. 이전 버전과 smoke 디렉터리는 이력과 실행 확인용입니다.
- checkpoints/, evaluation/, training_samples/는 실험 재개·재현에 쓰므로 경로와 파일명을 유지합니다.
