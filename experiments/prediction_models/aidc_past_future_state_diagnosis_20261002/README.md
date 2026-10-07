# AIDC 과거·미래 상태 진단

기존 개발 leave-day-out OOF prediction과 원본 완료 bin만 사용하는 사후 진단이다. 모델을 로드하거나 학습·inference하지 않는다. 원본 산출물은 보존한다. 공통 sample key는 `(run_id, segment, bin)`이다.

5초 primary와 1초 sensitivity의 range threshold는 각 fold의 기존 cap300 training anchor 전체에 대한 비가중 linear q50이다. 미래 endpoint q50·q90도 같은 training에서 복원하고 저장값을 검증한다. λ=0은 공통 P 한 번, A는 원래 예측, R/RM은 λ=0.25·0.5·0.75·1을 모두 보고한다. 상태별 계수 선택을 하지 않는다.

프로젝트 루트 `/home/work/AIDC`에서 실행:

```bash
python experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/pipeline.py --smoke --output-dir experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/outputs/smoke_v2
python experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/manage.py launch --output-dir experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/outputs/full_v2
python experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/manage.py status --output-dir experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/outputs/full_v2
python experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/manage.py launch --resume --output-dir experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/outputs/full_v2
python experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/manage.py launch --report-only --output-dir experiments/prediction_models/aidc_past_future_state_diagnosis_20261002/outputs/full_v2
```

Launcher는 `nohup`, Python `-u`, stdin `/dev/null`, 새 process session을 사용한다. PID와 `/proc` start_ticks·명령을 대조해 상태를 확인한다. Launcher lock과 analysis lock이 중복 실행을 방지한다. 같은 input/code/config hash만 resume를 허용한다. Resume는 결정적인 분석을 재계산하며 report-only는 집계 checkpoint hash를 검사한 뒤 그림·보고서만 다시 생성한다. 코드가 바뀌면 새 output directory가 필요하다.

`anchor_states.parquet`에는 과거 25개와 미래 5개 power·timestamp, 저장 prediction, range·threshold·상태가 들어 있다. Original bin은 Arrow run_id filter로 개발 run만 materialize한다. 전체 원본 파일의 hash 계산은 byte 단위이며 보류 날짜 행을 파싱하지 않는다. Bin 누락 시 영향 key와 offset을 오류로 기록하고 중단하며 anchor를 임의 제외하지 않는다.

상태별 성능은 비어 있지 않은 run 간 동일 가중 평균이다. p95는 run-cell n≥20, p99는 n≥100; n<30은 small_n이다. 빈 성능은 N/A, 빈 gain 기여도는 0이다. Contribution 분모는 각 run의 전체 evaluation anchor 수이고 여섯 cell 합은 전체 gain과 같다. 방향 일치율은 큰 endpoint 변화에서만 계산하며 Persistence의 0 변화는 불일치다.

`metrics_summary_state.csv`에는 whole·여섯 cell·큰 변화 상승/하강·first departure를 모두 기록한다. `metrics_by_day_state.csv`와 `metrics_per_run_state.csv`에서 날짜별·run별 값을 확인한다. `state_counts_per_run.csv`는 cell 비중과 small_n, `state_composition_*.csv`는 날짜·hardware/workload 구성이다. `gain_contributions.csv`, `metadata_effect.csv`, `sensitivity_comparison.csv`, `first_departure_metrics.csv`를 별도로 저장한다. `plotting_data/`의 CSV는 각 PNG·SVG에 사용한 수치다. 사례는 각 primary cell의 사전순 첫 key로 고정한다.

`verification.json`은 provenance·시간 정렬·공통 표본·threshold·membership·endpoint λ·기여도·기존 shrinkage 전체 수치 일치를 확인한다. `render_verification.json`은 그림 파일과 canvas 안의 텍스트를 점검한다. 한국어 보고서는 `report_ko.md`, `report_ko.html`이다. 실행 로그·PID·시작 시각·종료 코드는 output에 저장하며 `timing.json`은 전체 실행, `report_only_timing.json`은 그림·보고서 재생성 시간이다.

Bootstrap·유의성 검정·gate·새 모델·추가 threshold 탐색·보류 평가·제어 시뮬레이션은 수행하지 않는다. 개발 자료의 사후 결과이며 독립 일반화나 물리적 전이 시작 시각 예측을 입증하지 않는다.
