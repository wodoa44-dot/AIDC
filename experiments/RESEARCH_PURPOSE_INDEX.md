# 연구 목적별 실험 정리 대응표

기존 공식 실험명과 폴더 basename을 보존했다. 실행 코드의 절대 경로·폴더 깊이 의존성과 checkpoint identity를 유지하기 위해 7개 실험 모두 원래 위치를 canonical 경로로 유지하고, 연구 목적별 상위 폴더에서 상대 symlink로 접근한다. 실제 이동은 0건이므로 기존 경로에 새 호환 symlink를 만들 필요가 없으며, 기존 분류 링크도 그대로 유지한다.

기존 `experiments/README.md`와 `classification_manifest.json`은 일부 진단의 입력 해시에 포함되므로 수정하지 않았다. 이번 분류는 `research_purpose_manifest.json`에 별도로 기록했다. 기존 산출물의 이름·내용·provenance는 보존하며, 실행 상태나 연구 결과를 재평가하지 않는다.

| 기존 실험명 | 연구 목적 | 기존 경로 | 정리 후 canonical 경로 | 목적별 접근 경로 | 실제 이동 / 링크 분류 / 유지 여부 |
|---|---|---|---|---|---|
| aidc_grid_metadata_20260929 | 전력 history·metadata 예측 및 독립 tuning | `experiments/aidc_grid_metadata_20260929` | `experiments/aidc_grid_metadata_20260929` | [experiments/metadata_forecasting/aidc_grid_metadata_20260929](metadata_forecasting/aidc_grid_metadata_20260929/) | 링크 분류 · 원본 유지 |
| aidc_forecast_pilot_20261001 | 절대·residual target와 history 길이의 소규모 예측 비교 | `experiments/prediction_models/aidc_forecast_pilot_20261001` | `experiments/prediction_models/aidc_forecast_pilot_20261001` | [experiments/forecast_improvement/aidc_forecast_pilot_20261001](forecast_improvement/aidc_forecast_pilot_20261001/) | 링크 분류 · 원본 유지 |
| aidc_residual_metadata_followup_20261001 | residual·metadata 및 training cap 후속 개발 비교 | `experiments/prediction_models/aidc_residual_metadata_followup_20261001` | `experiments/prediction_models/aidc_residual_metadata_followup_20261001` | [experiments/forecast_improvement/aidc_residual_metadata_followup_20261001](forecast_improvement/aidc_residual_metadata_followup_20261001/) | 링크 분류 · 원본 유지 |
| aidc_residual_shrinkage_diagnosis_20261001 | 저장 OOF 예측의 residual shrinkage 오차 분해 | `experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001` | `experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001` | [experiments/forecast_diagnosis/aidc_residual_shrinkage_diagnosis_20261001](forecast_diagnosis/aidc_residual_shrinkage_diagnosis_20261001/) | 링크 분류 · 원본 유지 |
| aidc_past_future_state_diagnosis_20261002 | 과거·미래 상태별 저장 예측 오차와 gain 진단 | `experiments/prediction_models/aidc_past_future_state_diagnosis_20261002` | `experiments/prediction_models/aidc_past_future_state_diagnosis_20261002` | [experiments/forecast_diagnosis/aidc_past_future_state_diagnosis_20261002](forecast_diagnosis/aidc_past_future_state_diagnosis_20261002/) | 링크 분류 · 원본 유지 |
| aidc_control_fair_budget_20260930 | 장치별 동일 예산 제어 비교와 nested 평가 | `experiments/aidc_control_fair_budget_20260930` | `experiments/aidc_control_fair_budget_20260930` | [experiments/control_evaluation/aidc_control_fair_budget_20260930](control_evaluation/aidc_control_fair_budget_20260930/) | 링크 분류 · 원본 유지 |
| aidc_forecast_diagnosis_20261001 | 예측·제어 통합 진단 및 endpoint·trajectory MPC 비교 | `experiments/aidc_forecast_diagnosis_20261001` | `experiments/aidc_forecast_diagnosis_20261001` | [experiments/forecast_control_diagnosis/aidc_forecast_diagnosis_20261001](forecast_control_diagnosis/aidc_forecast_diagnosis_20261001/) | 링크 분류 · 원본 유지 |

## 보존 범위

- `nested`, `outputs/full_v2`, `smoke` 등 기존 내부 이름과 구조를 유지한다. 하위 결과나 cache를 별도 실험으로 재명명하지 않는다.
- 기존 `experiment_id`와 코드·config·모델·결과 파일을 변경하지 않는다.
- 기존 `prediction_models/`, `control_models/`의 분류와 링크를 유지한다.
- 링크에서 파일을 편집하면 같은 canonical 원본에 반영된다. 새 실행은 원본 README의 실행 위치·설정·출력 경로 규칙을 따른다.

## 검증

검증 결과는 [research_purpose_verification.json](research_purpose_verification.json)에 기록한다.

- **기존 실험명·폴더 basename 변경 0건**
- 실험 7개: 실제 이동 0건, 링크 분류 7건, 원본 위치 유지 7건.
- 목적별 링크 7개가 동일 canonical 원본으로 해석되며 기존 README의 내용 또는 부재를 그대로 유지한다.
- 기존 분류 링크 9개가 유효하며 원본 대상을 유지한다.
- 원본 내부 44,520개 항목의 경로·형식·크기·mtime·inode·링크 대상 변경 0건.
- 코드·config 201개 파일의 SHA256 변경 0건. 기존 experiment_id를 보존했다.
- 대용량 모델·결과는 파일 속성으로 확인했으며 전체 내용 재해시나 실험 재실행은 수행하지 않았다.
