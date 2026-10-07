# AIDC 실험 분류

예측 모델은 `prediction_models/`, 제어 모델과 예측·제어 통합 진단은 `control_models/`에서 찾는다. 분류 폴더는 기존 파일을 가리키는 상대 심볼릭 링크다. 원본 코드·설정·결과를 이동하거나 복제하지 않아 기존 경로와 checkpoint provenance를 유지한다.

| 분류 | 경로 | 내용 |
|---|---|---|
| 예측 | [metadata 예측 실험](prediction_models/aidc_grid_metadata_20260929/) | P2 history 및 P4 metadata 모델, quantile 학습, OOF 평가, 독립 tuning |
| 예측 | [nested predictor cache](prediction_models/nested_predictor_cache/) | 최신 leave-day-out 및 inner grouped 예측 모델과 prediction |
| 예측 | [endpoint 진단 prediction](prediction_models/diagnosis_endpoint_predictions/) | 최신 split endpoint 예측 평가용 cache |
| 예측 | [trajectory 진단 prediction](prediction_models/diagnosis_trajectory_predictions/) | horizon별 q50 모델과 prediction |
| 제어 | [동일 예산 제어 실험](control_models/aidc_control_fair_budget_20260930/) | 장치별 예산, 반응형·예측형 제어, nested tuning |
| 제어 및 통합 진단 | [예측·제어 최종 진단](control_models/aidc_forecast_diagnosis_20261001/) | 공통 controller core, MPC, 입력 교체·weight·trajectory 비교 |
| 제어 | [기존 E1 코드](control_models/legacy_e1/) | 예측 프로젝트에 포함된 PCC proxy 제어와 장치·주파수 민감도 코드 |

## 주요 결과

- [예측 독립 tuning 집계](prediction_models/aidc_grid_metadata_20260929/outputs/full_20260928T180455Z/independent_primary_v1/aggregate_metrics.csv)
- [최신 nested 제어 보고서](control_models/aidc_control_fair_budget_20260930/nested/outputs/full_v2/report_ko.md)
- [전체 예측·제어 진단 보고서](control_models/aidc_forecast_diagnosis_20261001/outputs/full/report_ko.md)
- [제어 활용 방식 후속 보고서](control_models/aidc_forecast_diagnosis_20261001/outputs/controller_review_v1/report_ko.md)

## 사용 방법

분류 폴더는 탐색과 파일 열기에 사용한다. 기존 README에 있는 실행·중지·재개 명령은 원래 경로로 실행한다. 같은 파일이 두 위치에서 보이더라도 링크 대상은 하나이며 편집도 같은 원본에 반영된다. 링크로 연결된 개별 제어 스크립트는 원래 프로젝트의 실행 환경과 설정을 사용한다.

metadata 프로젝트에는 초기 E1 제어 결과도 포함되어 있다. 통합 실험의 코드·모델·결과를 잘라 옮기면 실행과 provenance가 깨질 수 있어 실험 단위를 유지하고 필요한 영역을 추가 링크했다. `classification_manifest.json`에는 분류 경로와 원본 경로를 기록했다.
