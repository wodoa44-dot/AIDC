AIDC 예측모델 개선 가능성을 확인하는 소규모 pilot 실험을 구현하고 실행하라.

목적은 적은 학습 데이터로 아래 세 가지를 확인하는 것이다.

1. 기존 q50 예측은 어디에서 Persistence보다 나쁜가? Quantile 정렬이 영향을 주는가?
2. 절대 전력 대신 변화량을 예측하거나 history를 늘리면 개선되는가?
3. 선택된 history-only 구성에 metadata를 추가하면 이점이 있는가?

이번에는 **예측모델만 평가한다. 제어·MPC·저장장치·계통 시뮬레이션은 실행하지 마라.** 새로운 모델 계열이나 대규모 hyperparameter 탐색으로 확장하지 마라.

계획만 제시하지 말고, 자료 확인 → 표본 추출 → smoke → nohup pilot 실행 → 집계·시각화·한국어 보고서까지 수행하라.

## 1. 실제 프로젝트 경로 확인과 작업 폴더

현재 workspace에서 다음 구조를 가진 프로젝트 root를 확인하라. `/home/work/AIDC`를 검증 없이 하드코딩하지 마라.

확인된 프로젝트 상대 경로:
- `experiments/aidc_grid_metadata_20260929`
- `experiments/aidc_control_fair_budget_20260930`
- `experiments/aidc_forecast_diagnosis_20261001`
- `experiments/control_models`
- `experiments/prediction_models`
- `experiments/classification_manifest.json`
- `experiments/README.md`

AGENTS.md와 git status를 확인하고, `experiments/README.md` 및 `classification_manifest.json`을 읽어 현재 폴더 분류와 기존 코드 위치를 파악하라. 위 목록 중 실제 존재하지 않는 항목은 기록하라.

### 새 실험 위치

이번 작업은 예측모델 전용이므로 다음 위치에 생성한다.

`<PROJECT_ROOT>/experiments/prediction_models/aidc_forecast_pilot_20261001`

이미 존재하면 사용자 변경과 기존 산출물을 확인하고, 호환되는 checkpoint만 재사용한다. 실행 결과는 다음처럼 별도 디렉터리에 저장한다.

`<PILOT_ROOT>/outputs/<run_id>/`

### 기존 자료 참조

다음 위치를 우선 확인하라.

- 예측 feature·학습 코드:
  `<PROJECT_ROOT>/experiments/aidc_grid_metadata_20260929`

- 날짜 group·기존 nested 모델:
  `<PROJECT_ROOT>/experiments/aidc_control_fair_budget_20260930/nested`

- 예측 진단·저장 prediction:
  `<PROJECT_ROOT>/experiments/aidc_forecast_diagnosis_20261001`

`nested`를 포함한 하위 경로는 실제 존재 여부를 확인한다. 자료가 이동됐다면 README와 classification manifest의 기록을 우선 따라가고, 필요한 파일명·함수명을 프로젝트 내부에서 검색해 찾는다.

기존 제어 실험 폴더에서는 예측 관련 자료와 grouping 정보만 읽는다. 제어·MPC·저장장치 실험은 실행하지 마라.

### 경로와 원본 보존 원칙

- 기존 실험·데이터·모델을 이동하거나 덮어쓰지 않는다.
- `control_models`는 수정하지 않는다.
- 기존 standardized data와 모델은 읽기 전용으로 참조한다.
- 파일명만 같다는 이유로 재사용하지 말고 split·target·feature·학습 이력을 확인한다.
- 경로를 찾지 못하면 임의로 만들거나 다른 결과로 대체하지 말고 해당 자료의 부재를 기록한다.
- 설정 파일에는 프로젝트 상대 경로를 사용하고, 실행 시 확정된 절대 경로를 `resolved_config.yaml`에 함께 저장한다.
- nohup 실행에는 확인된 작업 디렉터리·Python 실행 파일·config·로그의 절대 경로를 사용한다.

학습 시작 전에 실제로 사용할 프로젝트 root, pilot root, 데이터 경로, 참조 모델 경로, 출력 경로를 로그에 표시하라.
## 2. 고정 실험 설정

- 입력 bin: 200 ms
- Horizon: 1초 하나
- 발행 시각: 최신 완료 bin의 끝 t
- Target: `(t+0.8, t+1.0]`의 200 ms 평균 GPU 합산 전력
- CPU·냉각·시설 전력 제외
- 모델: 기존 LightGBM quantile regression
- 새로 학습할 quantile: q50 하나
- Seed: 17 하나
- Trees: 100
- Leaves: 15
- Learning rate: 0.07
- Minimum child samples: 30
- L1/L2: 0/0
- Row/feature subsampling: 사용하지 않음
- Early stopping: 사용하지 않음
- History 후보: 1초·5초·10초
- CPU threads: 최대 4
- 모델 학습: 순차 실행
- 추가 hyperparameter 탐색: 없음

실제 API에 대응하는 parameter와 나머지 기본값을 resolved config에 기록한다.

## 3. 날짜 기반 분할

중복 제거된 기존 32개 run을 모두 후보로 삼고 유효성 조건으로만 제외한다.

기존 group manifest와 대조해 다음 날짜를 확인한다.

개발 날짜:

- 2025-09-16
- 2025-09-17
- 2025-09-20

보류 평가 날짜:

- 2025-09-22

개발 validation:

- Fold 0: 09-17·09-20 학습, 09-16 validation
- Fold 1: 09-16·09-20 학습, 09-17 validation
- Fold 2: 09-16·09-17 학습, 09-20 validation

최종 pilot fit은 개발 날짜 3개로 학습하고 보류 날짜에서 평가한다.

실제 날짜가 다르면 조용히 대체하지 말고 불일치를 기록하라. 분할을 성능에 맞춰 바꾸지 마라.

날짜 grouping은 독립 수집 session이 확인된 grouping이 아니다. 날짜와 hardware/workload 교락 및 기존 평가셋 재사용이라는 한계를 보고한다.

Run ID·파일명·수집일은 모델 feature로 사용하지 마라.

## 4. 유효 anchor 정의

Anchor t는 다음을 모두 만족해야 한다.

1. 완료된 200 ms bin의 끝 시각이다.
2. 같은 연속 segment에서 t까지 10초, 즉 50개 완료 bin의 전력 이력이 있다.
3. t+1초에 끝나는 target bin이 있다.
4. History부터 target까지 200 ms 간격으로 연속한다.
5. 필요한 전력 bin이 기존 coverage·결측 기준을 통과한다.
6. Feature가 다른 run·segment나 미래 전력을 참조하지 않는다.

결측·단절 구간을 보간해 anchor를 늘리지 마라.

기존 feature 구현이 추가 관측을 요구하면 필요한 과거 길이를 명시하고, 모든 비교 모델에 동일한 공통 유효 anchor를 적용한다.

1·5·10초 history 비교는 10초 history가 가능한 공통 anchor에서 수행한다. 이 조건으로 짧은 구간이 제외되는 한계를 기록한다.

## 5. 표본을 얼마나, 어떻게 추출할 것인가

### 5.1 최소 1초 간격 후보 생성

각 segment에서 유효 anchor를 시간순 정렬한다.

첫 유효 anchor를 선택하고, 직전에 선택한 anchor에서 최소 1초가 지난 첫 유효 anchor를 순차 선택한다.

전력 크기·target·예측오차와 무관하게 적용한다.

이 간격은 인접 표본 중복과 계산량을 줄이기 위한 것이다. 10초 history가 겹치므로 독립 표본이라고 주장하지 마라.

### 5.2 Training: run당 최대 300개

Run별 1초 간격 후보가 300개 이하면 모두 사용한다.

300개를 초과하면:

1. 후보를 시간순으로 정렬한다.
2. 후보 수가 최대한 같도록 연속 인덱스 구간 10개로 나눈다.
3. 각 구간에서 30개씩 비복원 추출한다.
4. Seed는 전역 seed 17과 run_id의 SHA256에서 안정적으로 생성한다.
5. 추출 결과를 timestamp 순으로 정렬한다.

Python 기본 hash()는 사용하지 마라.

같은 run이 여러 fold의 training에 등장하면 동일한 anchor를 사용한다. 모델·target·history마다 다시 추출하지 마라.

### 5.3 Validation·평가: run당 최대 600개

같은 1초 간격 후보에서 선택한다.

후보가 600개 이하면 모두 사용한다.

600개를 초과하면:

1. 시간순 후보를 10개 연속 인덱스 구간으로 나눈다.
2. 각 구간에서 60개를 인덱스 기준 등간격 선택한다.
3. 중복이 없는지 확인한다.

평가 표본은 모든 모델에 동일하다. Target 값이나 모델 성능을 보고 추출하지 마라.

### 5.4 짧은 run

1초 간격 후보가 30개 미만인 run은 primary 학습·평가에서 제외하고 이유를 기록한다.

나머지 짧은 run은 가용 표본을 그대로 사용한다. 행을 복제해 quota를 채우지 마라.

최소 30개 기준은 pilot 설계값이며 통계적 충분성을 보장하지 않는다. 평가 run이 빠져 특정 fold를 구성할 수 없으면 이를 명시하고 무작위 row split으로 대체하지 마라.

### 5.5 Feature 계산 순서

연속 시계열에서 필요한 history를 읽어 feature를 계산한 후 anchor를 선택하거나, 미리 선택된 anchor의 연속 원본 history에서 feature를 계산하라.

**샘플링된 anchor끼리 lag·rolling을 계산하지 마라.**

같은 anchor에서 1초 모델은 1초 이내, 5초 모델은 5초 이내, 10초 모델은 10초 이내 정보만 사용한다.

### 5.6 Run별 학습 가중치

각 fit의 training run 수를 R, 전체 행 수를 N, run r의 행 수를 n_r라 하면:

`sample_weight = N / (R × n_r)`

Run별 총 가중치가 같고 전체 평균 가중치가 1인지 확인한다.

모든 비교 모델에 같은 가중치를 적용한다. 전이 oversampling이나 label 기반 가중치는 이번에는 추가하지 마라.

## 6. 실험 1: 기존 q50 및 quantile 정렬 감사

새 학습 전에 기존 저장 결과에서 다음을 비교한다.

- Persistence
- 원래 q0.5 모델 출력
- 15개 quantile 정렬 후 중앙값

같은 sample key·target·모델 provenance가 확인된 표본에서만 비교한다.

기록:

- MAE, signed bias, p95 absolute error
- 정렬 전후 q50이 달라진 비율
- q50 변화량
- Run별 개선·악화

원래 출력이 없지만 15개 저장 모델이 있으면 선택한 평가 anchor에서만 inference한다.

기존 모델이 평가 run을 학습에 사용했으면 일반화 평가가 아니라 정렬 동작 감사로만 표시한다.

모델·출력을 확보할 수 없으면 해당 항목을 unavailable로 기록하고 나머지 실험을 계속한다. **이 감사를 위해 15개 quantile을 새로 학습하지 마라.**

기존 감사 결과와 새 pilot 결과는 별도 표로 제공한다.

## 7. 실험 2: Target × History

History-only 입력으로 다음 6개 구성을 비교한다.

- Absolute × 1초
- Absolute × 5초
- Absolute × 10초
- Residual × 1초
- Residual × 5초
- Residual × 10초

Absolute target:
`future_power`

Residual target:
`future_power - current_completed_bin_power`

Residual 최종 예측:
`current_completed_bin_power + predicted_change`

Absolute/residual 비교는 target 이외의 feature·anchor·weight·학습 설정을 동일하게 한다.

Power-history feature는 기존 정의를 우선 사용한다.

- 현재 전력
- History 범위 내 lag
- 인과적 이동평균·표준편차
- 과거 전력 차이

History별 feature 목록·개수를 저장한다. Centered rolling, 미래 보간, segment 경계 참조를 금지한다.

각 구성에 개발 3-fold validation을 수행한다.

학습량:
**6개 구성 × 3 folds = 18 fits**

Persistence는 현재 완료 bin 전력을 그대로 예측하며 같은 평가 anchor를 사용한다.

모델 선택:

1. 개발 OOF run-balanced MAE 최소.
2. 수치상 동률이면 run-balanced p95 absolute error 최소.
3. 이후 짧은 history, absolute target 순으로 결정한다.

Run별 지표를 먼저 산출한 뒤 run 간 동일 가중으로 집계한다. Fold 크기가 다른데 fold 평균을 동일 가중하여 run-balanced라고 부르지 마라.

보류 평가 날짜를 구성 선택에 사용하지 마라.

예측값 후처리는 모델별로 동일하게 적용한다. 0 하한을 사용하면 clipping 전후 성능과 발생 비율을 기록한다. Test 최대 전력을 상한으로 사용하지 마라.

## 8. 실험 3: Metadata 추가

실험 2에서 선택한 target/history를 고정한다.

동일 구성에서:

- History-only
- History + metadata

Metadata는 실제 존재하는 다음 항목을 사용한다.

- Hardware
- Workload
- Model size
- Image size
- Batch size
- Sequence length
- ZeRO stage

존재하지 않는 값을 추정해 채우지 마라. 실제 사용한 열과 자료 출처를 기록한다.

범주 encoding은 해당 training fold에서만 학습한다. Unseen 범주는 -1 등 명시된 동일 규칙으로 처리한다.

개발 3-fold에서 metadata 모델만 추가 학습한다. 선택된 history-only 모델의 결과는 재사용한다.

**추가 3 fits**로 제한한다.

항목별 ablation, shuffle, 별도 metadata tuning은 이번에 하지 마라.

개발 validation은 앞선 구성 선택에 이미 사용됐으므로 metadata 개발 결과를 탐색적으로 해석한다.

평가 자료의 unseen 범주 비율·결측률·상수 열을 보고한다.

## 9. 최종 pilot 평가

개발 날짜 3개로 다음을 학습한다.

1. Absolute + history 1초 + history-only.
2. 실험 2에서 선택한 history-only 구성.
3. 선택 구성 + metadata.

동일 구성이면 재사용한다. Persistence는 학습하지 않는다.

보류 날짜의 동일 anchor에서 모두 평가한다.

**최대 3 fits 추가. 전체 새 학습은 24 fits 이하.**

결과를 본 뒤 target·history·metadata 설정을 재선택하거나 학습 표본을 늘리지 마라.

## 10. 예측 성능 지표

주 지표:

- Run-balanced MAE

보조 지표:

- RMSE
- Signed bias = prediction − actual
- p95 absolute error
- p99 absolute error
- Persistence 대비 MAE 개선율
- Run별 개선·악화 수

표본이 100개 미만인 run의 p99는 N/A로 두고 p99 집계 run 수를 표시한다. 나머지 지표도 표본 수를 병기한다.

구간별로 정상·상승·하강·큰 endpoint 변화를 평가한다.

기존에 train-only로 정의된 threshold가 있으면 재사용하고 출처를 기록한다. 없다면 해당 training 표본의 `abs(future_power-current_power)`로 다음을 결정한다.

- 정상: training 절대 변화량의 q50 이하
- 상승: 변화량이 +q50 초과
- 하강: 변화량이 −q50 미만
- 큰 변화: 절대 변화량이 training q90 초과

큰 변화는 상승·하강과 중첩되는 보조 구간이다. Threshold가 0이거나 구간이 비면 그대로 보고하고 임의로 표본을 채우지 마라.

같은 fold의 모든 모델에 동일 threshold를 적용한다. 최종 평가는 개발 training에서 계산한 threshold를 고정한다.

이는 **endpoint 변화 구간**이며, 미래 1초 동안의 모든 전이·최대 ramp 평가가 아니다. 구간 label은 평가에만 사용하고 입력 feature로 넣지 마라.

이번에는 q50만 새로 학습하므로 새 모델의 CRPS·q10–q90 coverage를 산출한 것처럼 보고하지 마라.

Bootstrap·정식 유의성 검정은 하지 않는다. Run별·날짜별 차이와 표본 수를 보고한다.

## 11. Sampling 감사와 수치 검증

학습 전 저장:

- run_inventory.csv
- excluded_runs.csv
- train_anchors.parquet
- evaluation_anchors.parquet
- split_manifest.json
- sampling_summary.csv
- resolved_config.yaml

Anchor manifest 필드:

- run_id
- collection_day
- segment_id
- anchor_timestamp
- target_timestamp
- 시간 구간 번호
- sampling_role
- seed 또는 deterministic 추출 규칙

모든 모델이 같은 manifest를 참조한다.

실제 fit별 다음을 출력한다.

- Training/validation run 수와 행 수
- 날짜별 run 수
- 원본 bin·유효 anchor·1초 간격 후보 수
- 추출 행 수
- Feature 수
- 공통 anchor hash

Training에서 결정한 endpoint threshold로 sampling 전후 구간 구성비를 비교한다. 이 비교를 보고 표본을 다시 추출하지 마라. 보류 날짜의 label 기반 진단은 최종 평가 단계에서만 수행한다.

필수 검증:

- 날짜 train/validation/test 중복 없음
- 동일 anchor에서 모델별 target 일치
- Lag·rolling의 인과성
- Residual target 복원 정확성
- Persistence 구현
- Run weight 합계
- CSV·표·그래프의 수치 일치

## 12. 시각화와 보고서

표:

1. 날짜 split·포함/제외 run·실제 표본 수.
2. 기존 quantile 정렬 감사 또는 불가 사유.
3. Target × history 개발 성능.
4. 선택 구성의 metadata 추가 효과.
5. 보류 날짜 최종 평가.
6. Endpoint 변화 구간별 성능.

그래프:

- History 1/5/10초에 따른 MAE: absolute/residual 꺾은선.
- 동일 구성의 p95 오차 꺾은선.
- 모델별 Persistence 대비 개선율: 점 또는 막대 그래프.
- Run별 MAE 차이.
- 개선·중간·악화 사례의 실제 전력과 예측 시계열.

History는 실제 수치 간격을 사용한다. 순서 없는 모델명을 꺾은선으로 연결하지 마라.

시계열은 짧은 연속 구간에서 별도 inference해 그린다. 불연속 평가 anchor를 연속 파형처럼 연결하지 마라.

예측선을 target 시각에 배치한다. 사례 선택은 사후 설명용이며 tuning에 사용하지 않는다.

PNG·SVG와 plotting CSV를 저장한다. 그림의 축·단위·범례·한글 잘림을 확인한다.

한국어 보고서는 Markdown과 HTML로 작성하고, 각 표·그림에 짧은 해석을 붙인다.

## 13. 계산량과 nohup 실행

Training cap:
**각 training run 최대 300행.**

Validation/evaluation cap:
**각 run 최대 600행.**

최종 training은 개발 run 수 × 최대 300행이다. 실제 수를 로그에 명시하라.

전체:

- Target/history 개발: 18 fits
- Metadata 개발: 3 fits
- 최종 pilot: 최대 3 fits
- 총 24 fits 이하
- 별도 초소형 smoke: 최대 1 fit

Smoke는 100행 이하·5 trees로 실행해 데이터와 학습 경로만 점검한다. Smoke 결과를 성능표에 포함하지 않는다.

Smoke 통과 후 전체 pilot을 **nohup으로 백그라운드 실행**하라.

- stdin: /dev/null
- Python: unbuffered
- stdout/stderr: 실행별 pipeline.log
- PID·명령행·시작 시각 기록
- progress.json·completion_status.json 기록
- 종료 코드 저장
- 단계/fold별 checkpoint와 resume
- 입력·코드·config identity가 일치할 때만 재사용
- 동일 작업 중복 실행 방지
- 학습 없이 표·그래프만 재생성하는 모드 제공

Pipeline은 학습 후 집계·시각화·보고서까지 자동 수행한다.

실행 직후 프로세스와 초기 로그를 확인하고 PID·로그 확인·중지·재개 명령을 제공한다. 백그라운드 시작을 완료라고 표현하지 마라.

성능이 나쁘다는 이유로 데이터 수·trees·seed·후보 수를 자동 확대하지 마라.

## 14. 최종 답변

다음에 답하라.

1. 기존 q50에 정렬 영향이 있었는가?
2. Residual target이 유망한가?
3. History 증가가 유망한가?
4. Metadata 추가가 유망한가?
5. Persistence를 넘는가?
6. 평균과 큰 변화 구간이 함께 개선되는가?
7. 다음에 학습량을 늘려 확인할 구성은 무엇인가?

결론은 다음 중 근거에 맞게 작성한다.

- 확대 검증할 가치 있음.
- 조건별로 달라 추가 확인 필요.
- 이 pilot에서는 개선 근거 없음.

작은 학습량의 실패를 모델 계열 전체의 한계나 예측 불가능성으로 해석하지 마라. 보류 날짜 하나의 결과를 일반화하지 마라.

실제 학습 행 수·fit 수·시간, 완료·불가 항목, 핵심 표·그래프와 보고서 경로를 제시하라.

이번 세 기본 비교를 끝낸 뒤 자동으로 다른 모델이나 대규모 실험을 시작하지 마라.
