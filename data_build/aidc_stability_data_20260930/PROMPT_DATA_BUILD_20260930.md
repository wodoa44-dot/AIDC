AIDC 계통 안정화 연구용 데이터 구축 — Codex 실행 프롬프트
아래 지시를 따라 /home/work/AIDC의 기존 자료를 확인하고, 새로운 독립 폴더에 데이터 구축 코드를 구현하여 실행하라. 계획만 작성하지 말고 기존 데이터에서 생성 가능한 데이터는 실제 생성·검증하라. 실측 장비나 계통 모델이 없으면 해당 부분은 실행 가능한 수집기·설정·절차와 결측 사유까지 구현하고, 없는 관측값을 생성해서 완료로 표시하지 마라.
0. 연구 목적과 범위
목적은 AI 데이터센터의 부하·전환 예측과 metadata가 제어를 통해 계통 주파수·전압·진동 응답 개선에 기여하는지 평가할 데이터를 만드는 것이다. 연구는 다음 연결을 따른다.
1. GPU/Server 제어, 빠른 전력전자·저장장치 보상, ESS 에너지 관리, UPS 전환·재접속을 고려한다.
2. 부하 상태, 제어 응답, 서비스 성능, 저장장치 여유, 접속점과 계통 특성을 파악한다.
3. 미래 전력뿐 아니라 전환 시점·변화폭·ramp·지속시간·보상 에너지와 제어 응답을 다룬다.
4. metadata 유무에 따른 예측 성능과 동일 제어기에서의 계통 응답을 비교한다.
이번 작업은 데이터 구축이다. 대규모 예측모델 튜닝은 별도 실험으로 남긴다. 작은 학습 smoke test가 데이터 인터페이스 검증에 필요하면 실행하되 연구 성과로 제시하지 않는다.
전력 평활화나 낮은 CRPS만으로 계통 안정화를 입증하지 않는다. 계통의 취약 모드, P/Q 응답, 장치 제약, 성능 비용을 별도로 평가할 수 있게 구성한다. 학습과 추론은 각각 분석한다.
1. 경로·독립성·실행
- 프로젝트 기본 경로: /home/work/AIDC/data_build/aidc_stability_data_20260930/.
- 경로가 이미 존재하면 기존 실행 manifest로 동일 작업인지 확인하여 명시적 resume으로 재개하거나 _v02 등 새 경로를 만든다. 다른 작업을 덮어쓰지 않는다.
- /home/work/AIDC가 없으면 현재 작업공간의 AIDC 후보와 AGENTS.md를 확인한다. 원본이 없는 경우 경로를 날조하지 않고 구현과 실제 데이터 실행 상태를 구분한다.
- 기존 trace/dataset, 전처리, 예측 코드와 결과는 읽기 전용으로 참고한다. 재사용 코드는 출처와 변경사항을 기록하고 새 프로젝트에서 동작하도록 한다.
- 원본 데이터, 다른 실험, 모델 패키지, 실행 중 프로세스를 수정하지 않는다.
- 새 코드·설정·캐시·결과·로그는 새 폴더 내부에 저장한다. 원본 대용량 파일을 불필요하게 복사하지 않는다.
- Python CLI, README, 환경 의존성 목록, config snapshot, seed, 코드·데이터 hash, 버전과 UTC/KST 시작 시각을 제공한다.
- 기본 CPU thread budget은 전체 합계 최대 4로 시작하고 환경에 맞게 조정한다.
권장 구조:
aidc_stability_data_20260930/
  README.md
  configs/{existing,events,phase0,grid}.yaml
  src/aidc_data/
    inventory.py
    canonical.py
    quality.py
    metadata.py
    events.py
    forecasting.py
    splits.py
    phase0/{collect,workloads,controls,identify}.py
    grid/{adapter,scenarios,validate}.py
    export.py
    cli.py
  tests/
  scripts/{launch_existing.sh,status.sh,resume.sh}
  manifests/
  outputs/<build_id>/
2. 반드시 구분할 데이터 종류
종류	생성 방법	입증할 수 있는 범위
D1 기존 실측 부하 데이터	원시 trace의 검증·표준화	기록된 부하의 특성과 예측 가능성
D2 예측 학습 데이터	인과적 feature와 미래 target 생성	새로운 run에서 예측 성능·metadata 유용성
D3 제어·성능 실측 데이터	명령, 전력, workload 성능 동시 측정	해당 장비·조건의 제어 지연·응답·간섭
D4 계통 시뮬레이션 데이터	출처가 확인된 동적 모델에 입력	명시한 모델·조건에서의 계통 응답
D5 대상 시설 실측 데이터	실제 PCC·저장장치·UPS 계측	해당 시설에서의 실제 계통·설비 동작


모든 산출물에 evidence_type을 둔다: measured_existing, measured_new, derived, simulated, synthetic_test. derived에는 부모 데이터의 evidence_type과 lineage도 보존한다. 단순 합성 fixture는 테스트 전용으로 분리한다. D4를 D5로 표현하지 않는다.
3. 먼저 AIDC 전체 자료를 감사하라
rg --files로 파일·README·schema·config·기존 보고서를 탐색한다. 실제 필요한 원본을 읽고 다음을 확정한다.
- 데이터셋, 원본 파일, run/session/job/node/GPU 식별자, 반복 실험 관계.
- 측정 대상: GPU-only, node CPU+GPU, job aggregate, rack, facility, PCC 중 무엇인지.
- 단위, timestamp 정의, 시간대, sampling 간격 분포, 센서 종류·갱신·평균화 설명.
- 원시값/보간값/집계값 여부와 보간 방식. 원시 데이터로 돌아갈 수 있는지.
- 누락·중복·역순·gap·비정상 값·GPU 누락·파일 간 중복.
- GPU·모델·batch·병렬화 등의 실제 metadata와 해당 값의 출처.
- 제어 명령 로그·성능 로그·PDU/IPMI·PCC P/Q/V/f·ESS/UPS 상태·계통 모델의 존재 여부.
이전 정리는 highres_node 32개, nlr_training 41개, nlr_online_rate 200개 canonical run을 제시했다. 이는 검증할 참고값이며 맞추기 위한 목표가 아니다. README의 session 수와 canonical 파일 수는 다를 수 있다. 독립 실험 수를 따로 산정하라.
highres는 약 20 ms timestamp 간격이라고 알려져 있으나 NVML 센서의 실제 시간 대역폭은 별도 확인이 필요하다. nlr_online_rate의 1 ms grid는 약 0.2초 원자료의 보간일 수 있다. nlr_training도 전처리 이력을 확인한다. 보간된 1 ms를 실측 해상도로 취급하지 않는다.
출력: inventory.csv, source_lineage.csv, quality_audit.csv, capability_matrix.csv, missing_data_requirements.csv.
capability_matrix는 목표별 available, derivable, requires_measurement, requires_model, unavailable과 근거를 기록한다.
4. D1: 표준 시계열과 metadata
4.1 공통 키·단위
최소 공통 필드:
dataset_id, source_file, source_hash, session_id, split_group_id, run_id, segment_id, timestamp, elapsed_s, sample_index, node_id, gpu_uuid, measurement_boundary, evidence_type, quality_flags.
측정값은 가능한 항목만 저장한다:
gpu_power_w, gpu_energy_j, cpu_power_w, node_power_w, pcc_active_power_w, pcc_reactive_power_var, voltage_v, frequency_hz, utilization_pct, sm_clock_mhz, memory_clock_mhz, temperature_c, enforced_power_limit_w.
없는 값은 null과 결측 사유로 기록한다. GPU power를 facility/PCC power 컬럼에 복사하지 않는다. 합산 GPU 수·대상 목록·coverage를 저장하고 GPU가 빠진 합계를 완전한 node 합계처럼 취급하지 않는다.
4.2 metadata schema
정적 metadata 후보:
- GPU model/count, CPU/node 구성, VRAM, 제조사 전력 한계.
- workload family(train/inference/image generation 등), model identifier/architecture/parameter count, precision/quantization.
- batch/microbatch/gradient accumulation, sequence/input/output 길이 정의, image resolution.
- data/tensor/pipeline parallelism, ZeRO stage, GPU 배치·노드 수.
- power cap, clock 설정, firmware/driver/runtime, 수집 장비와 센서 모드.
동적 metadata 후보:
- 현재 compute/communication/checkpoint/evaluation/idle/prefill/decode 상태.
- 현재 step/token 진행량, queue 길이, 관측된 request arrival, active batch, 메모리·network·I/O 상태.
- 현재 제어 상태, cap/filler 명령, SoC/UPS 모드.
- 사전에 공개된 checkpoint·작업 시작 계획과 그 계획이 알려진 시각.
모든 metadata에 value, unit, source, extraction_method, confidence, valid_from, valid_to, available_at 또는 동등한 schema를 둔다. 파일명 파싱은 규칙과 원문을 보존한다. filename 전체·run_id·session_id는 예측 feature에서 제외한다.
실제 완료된 output token 수·미래 request rate·전체 run 통계·사후 복원 phase는 발행 시점 feature로 사용하지 않는다.
전력만으로 추정한 phase는 inferred_phase로 표시하고 실제 instrumented phase와 구분한다. metadata가 일정하면 효과를 식별할 수 없음을 보고한다.
4.3 인과적 집계
- 중복·역순을 처리하되 원본과 수정 이력을 보존한다. gap을 넘는 window는 분리한다.
- 기본 비교 해상도는 200 ms completed bin. 원자료가 실제로 지원하는 경우 50/100/500/1000 ms도 별도 export한다. 20 ms 원자료는 보존한다.
- 불규칙 간격에서 단순 행 평균을 쓰지 말고 측정 의미와 시간 가중·coverage 규칙을 명시한다. backward/forward 시간 의미가 불명확하면 보수적으로 처리한다.
- closed bin의 availability는 관측 도착·sensor window를 고려한다. 실제 도착 시각이 없으면 가정이라고 기록한다.
- 양방향 보간·centered smoothing·미래값을 쓰는 정렬은 온라인 feature에 금지한다. 오프라인 분석용 보간은 별도 dataset으로 격리한다.
- valid coverage 기본 후보 0.95와 짧은 carry-forward 허용 여부를 config로 명시한다. 센서 의미에 맞지 않으면 조정하고 이유를 기록한다.
- 값을 반복 보간해 더 높은 해상도를 만들지 않는다. upsampling을 사용한 시뮬레이션에는 새 관측 정보가 없음을 표시한다.
5. D2: 이벤트와 예측용 데이터
5.1 이벤트 정의
상승/하강, valley, high-load 지속, 반복 진동, 작업 시작·종료·checkpoint 후보를 만든다. 실제 로그로 확인된 원인과 전력 형태만 보고 붙인 후보를 구분한다.
두 검출기를 분리한다:
- online detector: 시각 t까지 이용 가능했던 값만 사용. detected_at을 기록한다.
- retrospective labeler: 미래를 사용해 onset/end를 확정할 수 있으나 label 생성에만 사용.
이벤트 테이블:
event_id, parent_session, run_id, segment_id, direction, onset_time, detected_at, end_time, baseline_w, extreme_w, delta_w, duration_s, max_ramp_w_per_s, energy_deviation_j, label_source, confidence, right_censored, threshold_version.
baseline·변화폭·회복 기준·최소 지속시간·hysteresis·병합 간격·ramp 계산 간격을 수식과 config로 정의한다. gap이나 run 끝에서 종료가 관측되지 않으면 right-censored로 둔다. 작은 전력 근처 상대오차 폭주를 방지한다.
고정 이벤트 민감도 분석 후보로 상대 변화폭 5/10/20%, 변화 시간 0.2/0.5/1/2초를 사용할 수 있다. 이는 계통의 허용 기준이 아니라 분석용 후보이다. 절대 W 문턱은 센서 잡음·측정 경계에 맞춰 정하고 학습 데이터에서 추정하는 값은 fold 안에서 산출한다.
5.2 Forecast target
기준 시각 t는 마지막 완료 bin과 정보 도착 시각을 명확히 구분한다.
- 기본 200 ms 해상도에서 h=0.2, 0.4, …, 2.0초의 전력을 생성한다. 1초/2초를 주요 비교점으로 둔다.
- target power는 (t+h-0.2초, t+h] 구간 평균인지 다른 정의인지 고정하고 저장한다.
- 추가 H=5/10/30초는 run 길이와 target coverage가 충분한 경우에만 생성하며 별도 track으로 관리한다.
- 각 H 내 상승/하강 event 발생, 첫 전환까지 시간, 최대 상승·하강 ramp와 변화폭.
- 신뢰할 수 있는 라벨이 있으면 valley/high-load 지속시간과 미래 부족·잉여 에너지.
에너지 라벨은 예를 들어 발행 시각에 정한 기준 p_ref에 대해 integral(max(P-p_ref,0) dt)와 integral(max(p_ref-P,0) dt)를 각각 J로 계산한다. p_ref는 기본 t의 관측 전력 또는 당시 알려진 reference schedule이다. 미래를 보고 유리한 기준을 선택하지 않는다. 이는 무제어 기준 부하의 에너지 편차이지 실제 저장장치 소요량을 확정하는 값은 아니다.
관측된 미래값이 label이며 quantile은 향후 모델 출력이다. 시점별 quantile을 붙여 joint trajectory나 에너지 quantile로 간주하지 않는다.
5.3 Feature와 split
전력 이력 후보 길이 1/2/5/10/30초를 지원한다. lag, causal slope, mean/std/min/max, 관측된 변화·유효 표본 수를 생성한다. 필요한 history가 없는 행은 해당 view에서 제외하며 제외 이유를 저장한다.
동일 target row_id로 다음 feature view를 제공한다:
1. current_power
2. power_history
3. power_history + static_metadata
4. power_history + static_metadata + available_dynamic_metadata
학습/추론, 측정 경계, 데이터셋은 구별하며 무조건 pooling하지 않는다. baseline과 metadata 모델은 동일 평가 행을 사용한다.
중복·분할 파일·동일 원본의 GPU별 채널·집계 해상도·simulated 변형을 같은 split_group_id로 묶는다. run 내 행 무작위 분할을 주 평가로 사용하지 않는다.
독립 group 수가 충분하면 outer 5 / inner 3 grouped split을 생성한다. 부족하면 fold 수를 낮추고 이유를 기록한다. 학습 fold 안에서만 scaling·범주 vocabulary·imputation·통계적 event threshold를 fit한다.
held-out GPU/model/workload 검증은 실제 group 수와 조합이 지원할 때 별도 stress split으로 만든다. 항상 등장하는 metadata 조합만 보고 일반화했다고 하지 않는다.
동일 run 내부 보조 평가에서는 history/target overlap을 고려한 purge/embargo를 둔다. 과거에 반복 사용한 holdout은 새 untouched test로 부르지 않는다.
출력: canonical_power.parquet, metadata.parquet, events.parquet, forecast_index.parquet, feature view별 parquet, targets.parquet, split_manifest.json, data_dictionary.md.
feature마다 max_available_at <= issuance_time 검증이 가능하도록 provenance를 둔다.
6. D3: Phase 0 제어 응답·성능 데이터
기존 passive trace로 명령의 인과적 효과를 확정할 수 없다. 실제 장비가 준비되지 않았어도 아래 collector, workload wrapper, experiment matrix, 식별 코드를 구현하라.
6.1 계측기
- NVML binding은 설치 버전과 공식 문서를 확인하여 구현한다. power/energy API 미지원은 명시적으로 처리한다.
- polling 목표 간격은 10/20/50/100 ms 비교 가능하게 한다. 10 ms polling이 10 ms 독립 센서 관측이라는 주장을 하지 않는다.
- 각 read의 monotonic start/end, wall-clock UTC, 실제 간격, API duration, 값, 오류를 저장한다. 반복되는 동일 값만으로 내부 sensor refresh를 확정하지 않는다.
- 누적 energy 지원 시 단위·reset·wrap을 검증하고 delta-energy를 별도 기록한다.
- node IPMI/PDU와 PCC meter가 있으면 별도 collector로 수집한다. 서로 다른 측정 경계와 지연을 보존한다. 없는 계측기는 더미 값 대신 disabled_reason으로 표시한다.
- 다중 host는 시계 동기화 방법·offset/uncertainty를 기록한다. alignment 오차보다 작은 d_act는 unresolved로 둔다.
- GPU 전력 합계·node 총전력·PCC 전력을 동일시하지 않는다. 비교 시 손실·CPU·냉각 범위 차이를 설명한다.
6.2 실측 matrix
먼저 GPU capability와 기존 workload·모델·checkpoint를 탐색하여 재사용 가능한 최소 실험을 정한다. 임의 대형 모델 다운로드나 기존 연구 workload 중단을 하지 않는다.
프로토콜은 아래 축을 단계적으로 지원한다:
- 학습: compute 중심, compute/communication 반복, instrumented checkpoint/evaluation/일시 정지.
- 추론: steady arrival, burst, 일시 무요청 후 복귀; arrival/queue/prefill/decode/TTFT/ITL/완료시각 기록.
- baseline, cap down/up, filler on/off, gradual release, 여러 GPU가 있는 경우 staggered release.
- cap은 장비의 허용 범위 안에서 default limit의 60/70/80/90/100% 후보를 clamp·deduplicate한다. 적용 cap과 실제 enforced limit을 모두 저장한다.
- filler는 저우선순위 stream의 행렬곱 크기·반복수·duty cycle로 5~6개 수준을 교정한다. 다른 process와 실행할 때 stream priority만으로 전역 우선순위를 보장한다고 하지 않는다. MPS 유무·동시 실행 방식·지원 여부를 기록한다.
- 모델·batch·GPU가 여러 종류면 지원되는 조합만 사용한다. 한 GPU로 다른 GPU 세대 데이터를 합성하지 않는다.
- 초기 pilot은 가능한 조건별 독립 3회 반복을 시작점으로 하되, 연속 표본을 반복 실험 수로 세지 않는다. trial 순서를 seed로 무작위화하고 warmup·cooldown·baseline을 둔다. 효과가 불안정하면 추가 반복 필요로 보고한다.
6.3 명령·응답·성능의 연결
commands.parquet에는 command_id, trial_id, intended/apply/return time, API latency, requested/enforced value, 성공·실패, 관련 GPU를 저장한다.
workload_events.parquet에는 phase의 actual_time과 available_at, step/token/request id, 계획과 실제 실행을 기록한다.
분석 항목:
- 관측 valley 깊이 Δ: baseline과 최소 전력의 차이. 측정 window·필터와 함께 보고한다.
- d_act_up/down: 명령 시각부터 식별된 전력 응답의 90%에 처음 도달해 사전 정의 dwell을 유지할 때까지의 시간. plateau 부재·잡음·자연 변동 때문에 식별 불가능하면 invalid와 이유를 남긴다.
- 시작 지연, settling time, overshoot, 실측 상승·하강 ramp, 명령 실패율.
- filler efficiency는 eta_fill = E_incremental / E_requested_compensation으로 명명한다. 이는 열역학 효율이 아니라 요청 보상 에너지의 달성 비율이다. 분자·분모·단위·window를 함께 저장하고 1로 clipping하지 않는다.
- E_incremental은 가능한 동일 workload/seed/phase의 no-filler paired baseline과 비교한다. filler로 주 작업 진행이 달라져 baseline 대응이 깨지는 경우 apparent gain과 attribution limitation을 기록한다.
- 학습 step time/throughput/완료시간, 추론 TTFT/ITL/p95·p99 latency/throughput, 주 작업 복귀 지연과 backlog/rebound.
- 전력 감소율만 보고 서비스 품질 보존을 선언하지 않는다. 같은 작업 완료량과 SLA를 비교한다.
- 장비·workload·제어별 분포와 독립 trial 기반 CI를 저장한다.
6.4 하드웨어 실행 조건
이번 기본 all 명령은 기존 데이터 구축·검증이며 물리적 GPU 제어를 실행하지 않는다. live collection은 별도 collect --live --config ... 진입점으로 구현한다.
실측 실행 설정에는 사용할 GPU UUID, 해당 GPU의 실험 사용 권한·점유 상태, workload, run duration, 허용 cap 범위, 온도·오류 종료 조건이 구체적으로 있어야 한다. 이미 제공된 사용 권한을 재요청하지 말고 환경 정보가 부족하면 그 정보만 명시한다.
다른 사용자의 process를 종료하지 않는다. 기존 cap/clock을 저장하고 정상 종료·예외·signal 시 복구한다. watchdog, 최대 실행시간, collector/workload 종료 처리를 구현한다. 보호·UPS·실제 계통 장비 설정 변경은 이 코드의 자동 실행 대상에 포함하지 않는다.
장비가 없으면 collector와 protocol 준비 완료, 실측 데이터 없음으로 보고한다. synthetic smoke test를 실측으로 제출하지 않는다.
7. D4/D5: 계통 검증에 필요한 데이터
7.1 시뮬레이션 모델
먼저 AIDC 내 기존 동적 계통 모델·benchmark·논문 재현 코드를 조사한다. 없으면 공개 원문·저자 코드·공식 문서에서 재현 가능한 모델을 확보한다. 모델 파일·라이선스·버전·출처·수식·파라미터를 기록한다. 다음은 참고 시작점이며 수치는 원문을 확인한다.
- https://arxiv.org/abs/2508.14318 — GPU/소프트웨어/저장장치 부하 완화.
- https://arxiv.org/abs/2505.16575 — IT·냉각·UPS·재접속 동적 모델.
- https://arxiv.org/abs/2602.01013 — GFM-BESS의 P/Q 지원과 독립운전 모의.
- https://arxiv.org/abs/2510.05437 — 동적 부하의 계통 위험과 조건별 응답.
- https://arxiv.org/abs/2609.04398 — 계통 취약 모드를 고려한 HESS MPC. 축약 주파수 모델의 한계를 보존한다.
- https://www.pnnl.gov/publications/measurement-adequacy-monitoring-data-center-oscillations — 계측 대역폭과 필터링.
출처 있는 swing+governor 모델을 초기 단계로 구현할 수 있다. 이 모델로는 aggregate frequency response만 평가하며 voltage/EMT/UPS 세부 안정도를 검증했다고 하지 않는다. V/Q·UPS·약계통 상호작용은 이를 표현하는 동적 모델을 확보한 경우에만 생성한다.
임의 식으로 voltage = 1 - k*power를 만들거나 정적 power flow를 동적 안정도 결과로 사용하지 않는다.
7.2 시나리오와 입력 provenance
- 실제 held-out trace replay, GPU 수/동기화 차이, 계통 조건, 장치 크기·지연·초기 SoC, 재접속 시나리오를 config로 분리한다.
- 단일 node trace의 MW 확대는 scaled_scenario로 표시하고 배율·동기화 가정을 보존한다. 데이터센터 실측이라고 부르지 않는다.
- 강계통/약계통은 해당 network 모델의 임피던스·단락용량 등으로 정의한다. 단순 주파수 모델에서는 저관성·고관성 등 실제로 표현한 축만 사용한다.
- 인위적 정현파/step/fault는 synthetic stress input으로 분리한다. 원래 데이터에 해당 현상이 있었다고 하지 않는다.
- 참고 파라미터, 실측 식별 파라미터, 가정 파라미터를 구별한다. 특정 논문의 수치를 우리 시설의 검증된 값으로 사용하지 않는다.
- 제어 없는 경우, 반응형 제어, 향후 예측 기반 제어 입력을 수용할 adapter를 만든다. 예측 모델이 없으면 반응형 또는 명시적인 perfect-information oracle만 실행하고 이를 학습 모델 결과로 기록하지 않는다.
- 같은 부모 trace의 확대·노이즈·계통 조건 변형은 독립 관측 run 수를 늘리지 않는다. 부모 split을 계승한다.
저장 필드:
scenario_id, parent_trace_id, split_group_id, model_version, solver, simulation_dt, control_dt, forecast_dt, scaling_factor, parameter_source, initial_state, disturbance_id.
시계열에는 지원되는 P_IT, P_PCC, Q_PCC, voltage, frequency, storage power/energy/SoC, controller command, converter state, UPS mode를 저장한다. 관측 feature와 simulator hidden state를 별도 분리한다.
SoC·P/Q 출력 한계·충방전 효율·비상 예비량·입출력 지연을 모델링한다. 같은 물리 배터리를 UPS와 ESS에 중복 배정하지 않는다.
7.3 검증
초기 평형, 전력·에너지 balance, 부호·단위, 제약 위반, 포화, SoC 고갈, 단순 step 응답을 확인한다. solver step을 절반으로 줄여 주요 지표의 차이를 비교하고 허용오차를 기록한다. 제어/예측 주기와 solver step을 혼동하지 않는다.
모드 대역 power, 주파수 최대편차·RoCoF·회복시간, 모델이 지원하면 전압 편차·회복시간을 저장한다. RoCoF·회복시간의 window와 기준을 명시한다. 강제 진동 감소를 고유 감쇠비 개선이라고 부르지 않는다.
D5 실제 PCC 자료가 없으면 schema·수집 요구사항·시간 동기화·필요 장비 목록만 제공한다. 실측 전압·주파수 파일에 시뮬레이션 값을 채우지 않는다.
8. 자동 검증·산출물
구체적인 위험을 확인하는 테스트를 구현한다:
- 미래 값을 바꿔도 과거 feature가 바뀌지 않는지.
- 동일 부모 session이 split을 넘지 않는지.
- target window·issuance·available_at 정렬과 horizon 단위.
- 보간 provenance, gap/coverage, 누락 GPU 합산 처리.
- right-censored 이벤트, onset/detected_at 구분, 적분값·ramp 계산.
- measured와 simulated 분리, 단위 변환, storage energy balance.
- resume 후 중복 key 없음, 입력·config·code 변경 시 cache 무효화.
- fake NVML로 collector 오류·미지원 API·복구 흐름을 검증하되 테스트 결과와 실측 결과를 구분.
필수 산출물:
1. inventory.csv, quality_audit.csv, capability_matrix.csv, missing_data_requirements.csv
2. D1/D2 parquet, metadata schema, event definitions, feature dictionary, split manifest
3. phase0_protocol.md, phase0_matrix.csv, collector·제어·식별 코드와 실측 여부
4. grid_model_provenance.json, grid_scenario_manifest.csv, 실행한 경우 D4 시계열·지표
5. dataset_card_ko.md: 측정 경계·해상도·독립성·누락·용도·제약·evidence level
6. validation.json, build_status.json, report_ko.md, 실행·재개 로그
7. 후속 Stage4 실험용 dataset_manifest.json과 loader 예제. 경로, schema/version/hash, 단위, split, target와 feature view를 제공하되 기존 실험을 수정하지 않는다.
간단한 검수 그림: raw/집계 power, sampling interval과 gap, 이벤트 onset/detected_at, 조건별 metadata coverage, 실측 존재 시 command→power→performance, 시뮬레이션 존재 시 P/Q/V/f와 SoC. 실제 산출 데이터만 그린다.
9. 실행과 완료 보고
CLI: audit, build-existing, label, export, validate, prepare-phase0, collect, identify, prepare-grid, simulate-grid, report, all, status, resume.
실행 순서:
1. audit 후 실제 사용 가능한 파일·항목 확정.
2. 각 데이터셋 소수 run smoke로 인과성·schema 검증.
3. 통과한 전체 기존 데이터 구축을 launch_existing.sh로 nohup 실행. 고유 build_id, PID, 로그, heartbeat, exit code와 완료 상태를 기록한다.
4. 수집기·실측 계획을 구현하고 장비 없는 부분도 준비 상태를 남긴다.
5. 재현 가능한 계통 모델이 확보되면 작은 시나리오부터 실제 실행·검증한다. 확보 실패 시 model dependency를 구체적으로 기록한다.
6. Stage4에서 바로 읽을 수 있는 manifest와 한국어 보고서 작성.
status는 completed, running, failed, blocked_hardware, blocked_source, blocked_model, not_applicable 등을 구분한다. 구현 완료와 데이터 생성 완료를 별도 필드로 둔다.
실험이 길면 정상 기동과 진행을 확인한 뒤 상태 확인 명령을 제공한다. 시작한 것만으로 완료라고 하지 않는다.
최종 응답은 다음을 실제 값으로 보고하라:
- 새 폴더와 실행·재개 명령.
- 데이터셋별 원본/독립 session/run/유효 window/event 수와 제외 사유.
- 생성된 target·metadata, 측정 해상도와 검증되지 않은 sensor bandwidth.
- 기존 데이터로 확보한 부분, 실측이 필요한 부분, 모델 가정으로 생성한 부분.
- 테스트 결과, 실행 중인 PID/로그, 미완료 항목.
- 예측·제어·계통 효과 각각에 대해 현재 이 데이터가 무엇을 검증할 수 있고 무엇을 검증할 수 없는지.
Metadata가 유용하다는 결론을 미리 정하지 않는다. 없는 metadata, actuator response, 실제 계통 개선을 합성하여 긍정적인 연구 결과를 만들지 않는다.