AIDC 4단계: 새 폴더에서 예측·metadata·협조 제어·계통 응답 실험 구현 및 실행
당신은 /home/work/AIDC 저장소에서 작업하는 Codex다. 아래 지시를 읽고 실제 코드 구현, 테스트, smoke, 본 실험 실행, 결과 집계까지 진행하라. 계획이나 코드 골격만 제시하고 끝내지 마라. 이 프롬프트는 독립 실행용이며 이전 대화나 보고서 첨부 없이도 수행할 수 있도록 작성되었다.
1. 연구 목적과 새 작업 경계
연구 목표는 워크로드 정보가 포함된 부하 예측을 GPU·전력전자 설비·ESS 협조 제어에 연결했을 때, 서비스 및 장치 제약 안에서 계통의 전압·주파수·진동 응답을 개선할 수 있는지 검증하는 것이다.
평균 CRPS 감소, GPU power smoothing, PCC ramp 감소, 계통 동특성 개선은 서로 다른 결과다. 각각 따로 평가한다. Metadata의 유용성을 미리 결론 내리지 않는다. 개선·무효·악화·판정 불충분을 모두 보고한다.
새 프로젝트를 다음 위치에 만든다.
/home/work/AIDC/experiments/aidc_grid_metadata_20260929/
실제 저장소 루트가 다르면 해당 루트 아래 동일한 상대 경로를 사용하고 기록한다. 이 폴더가 이미 존재하면 덮어쓰지 말고 _v02 등 새 폴더를 만든다. 이후 모든 코드·설정·cache·모델·로그·결과는 이 새 프로젝트에만 쓴다. 기존 trace/preprocessing, trace/predictability*, trace/model_200ms_1s_2s는 읽기 전용 참고 자료다. 기존 학습 코드에 새 실험을 덧붙이거나 과거 output을 최신 결과로 복사하지 마라.
AGENTS.md와 git status를 확인하고 사용자 변경을 보존한다. 기존 구현의 필요한 작은 함수는 출처·hash와 함께 새 프로젝트에 옮기거나 명시적 read-only adapter로 재사용한다. 원본 데이터는 외부 경로로 읽고 대량 복제하지 않는다. 실제 GPU cap·UPS·ESS·계통 장치에 명령을 보내지 않는다. 이번 작업은 offline 학습 및 시뮬레이션이다.
2. 이미 정리된 방법론
다음은 사용자와 정한 연구 설계다. 단순 GPU 전력 예측 연구로 축소하지 마라.
1. GPU·Server: 지원되는 power cap, floor, ramp, stop-delay, filler 등으로 변동을 1차 완화하되 처리량·latency 손실을 고려한다.
2. 전력전자 설비: 빠른 잔여 변동을 보상하고, 지원되는 인버터 기능으로 전압·무효전력 응답에 관여한다.
3. ESS: 지속적인 잔여 변동과 에너지 요구를 담당하며 SoC·비상 예비량을 유지한다.
4. UPS 이탈·재접속은 별도 외란 시나리오다. 기본 workload 변화 실험과 섞지 않는다.
예측은 상위 제어가 미래 변동에 대비하는 데 사용하고, 실측 feedback이 잔차를 수정한다. 200 ms 예측 주기를 전력전자 내부 제어 주기로 대입하지 마라. 물리 장치 하나가 여러 제어 기능을 담당할 때 용량을 중복 계산하지 않는다.
부하 예측모델, actuator response 모델, workload performance 모델, grid dynamics 모델을 분리한다. 무제어 trace로 학습한 predictor를 제어 이후 소비에 그대로 적용하면 분포가 달라질 수 있다. 기본 실험에서는 구분된 baseline demand와 제어 효과 모델을 사용하며, cap이 실행 시간을 바꾸는 효과까지 식별되지 않았다면 모델의 한계로 명시한다.
3. 데이터·모델·시뮬레이터 inventory
시작 시 다음을 실제 파일에서 점검한다.
- trace/dataset/High-resolution-AI-Data-Center-Training-Workloads-Dataset
- trace/dataset/nlr_312/extracted
- 기존 manifest, quality, metadata, 학습 설정 및 모델 artifact
- 저장소의 power-system network case, 동기기·부하·인버터 모델, controller, parameter source
- Cap/filler 실험, 처리량·latency, actuator delay, SoC 자료
과거 검토의 시작 정보이며 현재 inventory로 재확인할 내용:
- Highres node 32 canonical runs, 8 GPU 합산 power, CPU 제외, 기록 dt 약 20 ms. 전체 공개 데이터 72 session과 현재 node subset 32개를 혼동하지 않는다.
- NLR training 41개와 online-rate 200개가 과거 선택 범위다. NLR은 CPU+GPU이며 training은 job 전체 합이다. Highres와 동일 target으로 pooling하지 않는다.
- NLR online-rate 1 ms grid는 보간값이다. training 200 ms 집계에도 양쪽 관측 기반 보간이 사용됐다는 기존 audit이 있으므로 둘 다 원본 생성 코드를 확인한다.
- 기존 highres manifest는 GPU/workload/model을 보존하지만 model size·batch·sequence length·image size·ZeRO stage 등이 filename에만 있을 수 있다.
- 과거 events.py에는 centered smoothing과 recovery crossing 문제 가능성이 있다. events.csv는 10만 행까지만 export한 구현이 있다. 기존 event CSV를 완전하고 검증된 정답으로 재사용하지 않는다.
- 과거 manifest의 원본 row count와 분석 후 dt가 혼합되어 있을 수 있으므로 source/analysis timing을 분리한다.
업로드/기존 보고서에 기재된 값과 현재 확인한 값을 구분한다. canonical file이 다르다고 독립 session이라고 단정하지 말고 동일 원본 수집·실험 session을 group으로 묶는다.
capability_matrix.csv를 생성한다. 각 자료/모델의 가용성, 검증 수준, 출처, 가능한 실험, 불가능한 주장, 필요한 추가 자료를 기록한다. 자료가 없으면 0으로 채우지 않는다.
4. 전처리와 causal availability
Primary는 실제 사용할 수 있는 모든 highres node run이다. 임의로 사전순 첫 12개만 택하지 않는다. 제외는 품질/가용성 규칙과 근거를 남긴다.
200 ms 완료 bin을 기본으로 하고 bin의 오른쪽 끝과 실제 availability를 저장한다. 산술 sample mean과 time-weighted mean을 구분한다. 관측 dt, sensor 갱신·평균 window, 수집 지연을 구분한다. unknown은 그대로 둔다.
기존 비교를 위해 count coverage 80%, gap threshold 5×median dt를 초기값으로 두되 configuration에 공개한다. 결측·큰 gap·metadata identity 변경에서 segment/state를 재설정한다. longest-history 공통 eligible target을 사용해 paired 비교를 맞춘다. online feature에는 centered filter·양방향 interpolation·미래 관측으로 채운 값이 없어야 한다.
NLR은 raw asynchronous log로부터 각 시점에 도착한 샘플만 사용하는 causal reconstruction을 구현할 수 있으면 별도 track으로 평가한다. device별 age/staleness·합산 coverage를 기록하고 무제한 forward fill을 금지한다. 원시 timestamp가 수집 완료 시점인지도 확인한다. 불가능하면 primary에서 제외하고 정확한 이유를 남긴다. 보간 파일의 timestamp만 이동해 causal하다고 주장하지 않는다.
과거 보고서의 sensor dt와 장비 응답 시간을 실제 제어 지연으로 대입하지 않는다. GPU-only 시계열에서 시설 전력·Q·V·f를 관측한 것처럼 생성하지 않는다.
5. Targets와 features
첫 production 실험은 sampling 200 ms, horizon 0.2/0.4/0.6/0.8/1.0/1.2/1.4/1.6/1.8/2.0 s로 한다. 1/2 s는 사전에 정한 주요 평가 horizon이다. 중간 horizon은 궤적·전이 분석에 사용한다.
필수 targets:
1. 각 horizon 끝의 200 ms mean power quantiles.
2. horizon 내 상승/하강 event probability. endpoint ΔP와 path event를 구분한다.
3. 구간 내 최대 positive/negative ramp. 초기 ramp window τ=0.2 s, 보조 τ=1 s이며 h<τ에서는 N/A다. 과거·미래 bin을 포함하는 정의와 window endpoint를 명시한다.
상승·하강 크기에는 현재 기준 미래 최대·최소 편차도 저장한다. endpoint ΔP quantile은 power quantile에서 현재값을 빼서 계산하며 중복 모델의 가치가 없으면 새 모델을 만들지 않는다.
event threshold는 training-only |ΔP|의 80 percentile과 비영(positive) 최소폭 규칙을 초기 exploratory 정의로 사용할 수 있다. horizon별 학습 partition에서 산출 후 freeze한다. 물리 단위의 별도 scenario threshold도 보고하되 계통 허용치라고 부르지 않는다. 공통 test subset/threshold는 paired 모델 사이에 동일해야 한다. 임계값·노이즈 수준·정의 변경의 sensitivity를 남긴다.
시간·duration targets는 신뢰할 수 있는 event segmentation 이후 확장한다. 미발생·우측 검열을 0초로 넣지 않는다. 보상 에너지는 명시적 reference와 제어 정책이 있어야 정의 가능하므로 먼저 offline label/계산으로 만들고 충분한 근거가 있을 때만 직접 예측한다. marginal quantiles를 더해서 energy quantile이나 joint trajectory coverage를 만들지 마라.
Features: causal power lags/rolling stats/slope와 실제 확인된 static metadata. dynamic workload/control features는 발행 시점 가용 로그가 있을 때만 별도 조건으로 추가한다. constant GPU count의 효과를 주장하지 않는다. model size token을 parameter count로 임의 해석하지 않는다. GPU와 workload별 모델을 따로 만든 결과를 metadata 추가 효과라고 부르지 않는다.
6. 비교군·split·학습 예산
예측 비교군:
- P0: persistence point forecast; probabilistic baseline은 training-only residual distribution으로 별도 정의.
- P1: current-only learned model.
- P2: power history only.
- P3: history + 기존 수준 static metadata.
- P4: history + 확인된 확장 static metadata(model size/sequence/image size/ZeRO 등).
- P5: history + static + dynamic metadata, 실제 가용할 때만.
Primary metadata 비교는 P2 vs P4이며 P3는 기존 정보 대비 확장 효과를 보여준다. Primary는 dataset별 pooled 학습이고 cohort별 점수도 보고한다. 동일 조건 반복과 unseen configuration generalization을 별도 구분한다.
독립 group이 충분하면 outer GroupKFold 5, inner GroupKFold 3. group/cohort 분포를 확인하고 split manifest를 고정한다. 분할이 불가능하면 fold 수를 줄이고 이유를 기록한다. 이전에 관찰한 모든 데이터로 만드는 결과는 retrospective다. 과거 holdout 4개를 새 untouched test라고 부르지 않는다. within-run 60/20/20은 보조 분석이며 held-out-run을 대신할 수 없다.
학습 run당 최대 5,000개 deterministic temporal sample, run별 동일 총 weight를 초기 budget으로 쓴다. 샘플 index/weight를 저장하고 모든 paired 모델에 동일하게 적용한다. test는 공통 eligible issuance를 200 ms마다 전부 평가한다. tuning·feature encoding·imputation·event thresholds는 outer test를 보지 않는다.
기존 lightweight reference의 문서상 설정은 trees=100, leaves=15, lr=0.07, minimum leaf=30, L1/L2=0, history=1 s(+1 s)/5 s(+2 s)다. artifact에서 재확인하고 legacy reference로 재현한다. 이전 보고서의 40 trees 초기 모델과 혼동하지 않는다.
신규 quantile 모델 탐색 공간:
- n_estimators: [100,300,600]
- num_leaves: [7,15,31]
- learning_rate: [0.03,0.07]
- min_child_samples: [30,100,300]
- reg_lambda: [0,1,10], reg_alpha: 0
- history_s: [1,2,5,10,30]
- max_depth: -1, feature_fraction/colsample_bytree: 1, bagging/subsample: 1, subsample_freq: 0
- quantiles: [0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]
- search seed: 2026; evaluation seeds: [17,42,2026]. deterministic full-sample fits에서 seed가 동일 결과를 내도 인위적인 차이를 만들지 않는다.
전수 탐색 대신 baseline 포함 24개 unique candidate를 test 결과 확인 전에 생성·고정한다. 후보 parameter와 data row 수의 적합성을 검사한다. 모든 조건에 같은 후보 budget을 쓴다. 먼저 q10/q50/q90의 고정 weighted quantile loss로 후보를 screening하고 top 4를 full15 CRPS로 재평가한다. screening grid·기준을 저장하고 최종 점수와 혼동하지 않는다. seed 2026으로 tuning하고 선택 후 세 seed로 재적합한다. 신규 중간 horizon의 최적화 예산도 미리 manifest에 기록한다. early stopping은 기본 사용하지 않는다.
Controlled track: P2가 inner에서 선택한 feature-history·model capacity·sampling·iterations를 P3/P4에도 고정한다.
Independent track: P2/P3/P4 각각 동일 예산으로 선택한다.
전이 분류는 binary objective로 확률을 출력하며 training class weighting을 쓰면 calibration 영향을 평가한다. 기본 class weighting은 사용하지 않고 불균형 지표를 보고한다. 동일 후보 subset을 문서화하고 inner Brier score로 선택한다. event decision threshold는 inner data에서 고정된 false-alarm budget 아래 선택한다. 목적과 예산이 없는 임의 최적 F1 threshold를 test에 맞추지 마라.
Metadata 항목별 ablation은 primary 1/2 s에서 선택된 모델을 기반으로 실시한다. run-level metadata shuffle negative control 5회를 별도 실시하고 시간별 row shuffle로 static 구조를 깨지 않는다. 테스트 metadata는 정상 유지한다.
7. 평균·세부 오차·불확실성 평가
CRPS는 2×pinball을 quantile 축에서 수치 적분한 approximation으로 명시한다. 불균등 grid이므로 단순 평균하지 않는다. tail extension 정책을 고정하고 degenerate/uniform 분포 사례로 검증한다. crossing correction 전/후를 평가한다. compact3 score와 full15 CRPS를 동일 점수처럼 비교하지 않는다.
필수 지표:
- run-balanced CRPS, MAE, RMSE, signed bias
- absolute error p90/p95/p99, worst 1% 및 worst 5% 평균, max와 위치
- under/overprediction rate·magnitude
- quantile reliability, 80/90% coverage·width·interval score
- 상승/하강/안정 구간, 변화폭, cohort, run별 위 지표
- event Brier, PR-AUC, precision/recall, missed-event rate, false alarms/hour, lead time
- 최대 ramp 예측 오차, endpoint와 path event 차이
- 고정 W 오차 기준의 초과시간·최장 episode; 계통 안전 기준으로 부르지 않음
- feature/prediction/end-to-end CPU latency p50/p95/p99, 모델 크기
연속 event의 일대일 matching, detection dead time, matching tolerance, segment/censoring 규칙을 먼저 고정한다. 양성 event가 없는 fold의 PR-AUC는 N/A와 개수를 기록한다. timing error는 matched 사건만의 결과와 misses를 함께 제시한다.
집계는 row → run 내부 평균 → 같은 run의 seed 평균 → run 간 동일 가중. p99의 run 평균과 전체 row의 pooled p99를 별도 이름으로 저장한다. CI는 paired independent-group cluster bootstrap 2,000회(seed 2026). 기존 run 내부 10 s block CI를 unseen-run uncertainty로 쓰지 않는다. fixed-prediction CI는 재학습·선택 불확실성을 포함하지 않는다.
Meta gain=100*(1-score_meta/score_history). 분모 0은 N/A. 1/2 s primary 비교와 나머지 exploratory 분석을 구분한다. 개선된 cell만 선택하지 않는다.
모델 선택 두 버전:
1. mean_crps_selection: inner run-balanced CRPS 최소.
2. reliability_selection: inner 최소 CRPS 대비 1% 이내 후보에서 transition p95 absolute error 최소. 예측구간 coverage·width 악화도 함께 보고한다. 1%는 연구 설계값이며 최적/안전 기준이 아니다. 동률은 낮은 CRPS, 작은 모델 순으로 선택한다. 조건을 만족하는 후보가 없으면 명시적 fallback을 쓴다.
독립적으로 calibration을 추가할 경우 outer-training 내부 calibration partition을 분리하고 그 비용·표본 수를 보고한다. test를 보고 보정하지 마라.
8. 계층형 controller와 제어 효과
예측 점수에서 작업을 종료하지 마라. 다음 controller 비교를 구현한다.
- C0: no control
- C1: GPU-layer control only
- C2: causal reactive hierarchical control
- C3: 동일 계층·용량의 history-only predictive control
- C4: 동일 계층·용량의 metadata predictive control
- C5: 같은 actuator 제한을 갖는 perfect-future oracle, 진단용
P2/P4는 paired out-of-fold prediction을 사용한다. controller tuning은 outer-training 내부에서만 수행한다. rollout의 현재 feedback state와 predictor가 읽는 signal을 명시하여 baseline demand, controlled consumption, PCC power를 혼합하지 않는다. 제어된 GPU consumption이 필요하지만 counterfactual 성능/전력 모델이 없으면 해당 closed-loop branch를 validated로 표시하지 않는다.
설계 요구:
- baseline workload demand, GPU 제어 후 부하, 빠른 저장장치, ESS, PCC를 별도 signal로 유지한다.
- 지원되는 baseline controller는 causal ramp-limited reference와 잔차 분담으로 구현한다. 예측형은 동일한 제약 아래 receding-horizon으로 reference/보상 계획을 최적화한다. 첫 동작만 적용 후 재계획한다.
- 빠른/느린 보상 분담은 causal filter 또는 명시적 actuator 모델로 구현한다. future-aware zero-phase filter를 쓰지 않는다.
- 출력·slew·상/하향 delay·SoC·효율·비상 reserve·성능 제약을 적용한다.
- charge/discharge 동시 명령, 장치 용량 중복, energy 생성, feedback 없이 미래 true demand를 읽는 오류를 금지한다.
- P/Q capability와 전류 제한을 모델이 지원할 때 반영한다. 단순 P smoothing으로 Q/voltage control을 했다고 부르지 않는다.
- 기존 모델로 확인되지 않은 throughput/latency는 N/A다. passive trace를 cap으로 clipping한 계산은 idealized scenario일 뿐 실측 cap 효과가 아니다.
actuator 식별값이 없으면 delay 0/0.1/0.5 s, compensation capacity 0.1/0.3×training P95 load 등의 명시적 가정 시나리오로 제한된 sensitivity를 수행할 수 있다. baseline case를 사전에 고정하고 모든 조합을 무조건 전수 실행하지 않는다. η·효율·성능 비용을 관측값으로 꾸미지 않는다. 모델 provenance를 모든 결과에 연결한다.
9. 실제 계통 응답을 평가하는 경로
다음 evidence level을 반드시 구분한다.
- E0: 실제 trace의 예측 평가
- E1: 가정 또는 식별된 actuator에서 부하/PCC proxy smoothing 평가
- E2: 출처·파라미터·검증이 있는 benchmark grid의 동특성 simulation
- E3: 대상 시설·계통의 검증된 모델 또는 현장 실측
E1만으로 계통 안정화 입증 완료라고 선언하지 마라. E2도 실제 대상 계통의 현장 검증으로 표현하지 않는다.
먼저 저장소의 사용 가능한 동특성 simulator와 case를 조사한다. 있으면 read-only adapter로 연결한다. 없으면 접근 가능한 공식 예제/저자 benchmark에서 재현 가능한 동특성 case를 확보하여 별도 폴더에 출처·버전·license와 함께 구성한다. 정적 power flow만 수행하는 도구로 frequency dynamics나 oscillation damping을 계산했다고 주장하지 마라.
사용 가능한 모델이 없다면 외부 자료 없이 임의의 수식 Δf=kΔP, ΔV=kΔP를 만들어 계통 검증으로 대체하지 않는다. 공개 파라미터의 저차 frequency-response 모델은 제한된 frequency-only exploratory simulation으로 사용할 수 있지만, 전압·공간적 모드·GFM 안정도 검증은 미지원으로 표시한다. 확보할 수 없는 benchmark/engine/파라미터는 정확히 보고하고 나머지 구현·예측 실험은 완료한다.
계통 case별 필수 정보: base MVA, nominal f, topology, initial operating point, generation/control dynamics, load P/Q model, converter/control limits, integration step, 데이터센터 연결 모선, 초기화 방법, disturbance 정의. GPU W trace의 MW scaling·aggregation·동시성은 명시적 scenario다. 실제 대규모 시설 관측으로 부르지 않는다.
simulation step과 관측 cadence를 분리한다. 입력을 작은 solver step에서 유지하는 것은 새로운 고주파 관측 정보를 만드는 일이 아니다. 초기화는 각 controller의 운전 상태와 일관되게 하고, 초기 부하 mismatch가 의도치 않은 외란이 되지 않도록 한다.
기본 검증: baseline equilibrium, 알려진 load step 응답의 부호·정상값, power balance, step-size convergence, 모델이 지원하는 보호/제한 동작. 수치 발산과 물리 불안정을 구분한다. 모델 지원 범위에 맞춰 장치·계통 시간척도를 선택한다.
grid test scenarios:
- 동일 workload와 운전점에서 C0–C4 비교
- 부하 크기/반복성/동시성 변화
- 출처 있는 parameter로 구성한 강/약 접속 조건 또는 문서화된 impedance sensitivity
- UPS 외란·분리·재접속은 해당 모델이 실제로 지원할 때만 별도 실행
주파수: nadir/zenith, max |Δf|, 정의된 window의 RoCoF, settling/recovery time, bound exceedance duration.
전압: 대상/PCC 및 worst bus Vmin/Vmax, deviation, recovery, bound exceedance duration.
진동: forced response amplitude와 disturbance 종료 후 decay를 구분한다. 모델/신호가 지원할 때만 damping/mode 추정. 지속 강제응답의 PSD peak를 damping ratio로 부르지 않는다.
부하·설비: ramp 초과량/시간, peak-to-peak, storage power/energy, SoC/reserve 위반, filler energy, 관측·모델화된 성능 손실.
모든 case의 개선/악화를 보존한다. C3–C4는 예측 입력 외 동일 controller로 먼저 비교한다. 이후 controller 별도 tuning 결과는 독립 track으로 표시한다. 예측모델 selection에 outer grid test 응답을 사용하지 않는다.
10. 파일 구조와 실행 인터페이스
아래 구조를 기본으로 하되 중복은 줄인다.
aidc_grid_metadata_20260929/
  README.md
  configs/{smoke,full,control,grid}.yaml
  src/aidc_exp/
    inventory.py
    preprocessing.py
    features.py
    targets.py
    splits.py
    training.py
    evaluation.py
    control.py
    grid_adapter.py
    reporting.py
    cli.py
  tests/
  scripts/{launch_full.sh,status.sh,resume.sh}
  data_cache/
  outputs/<experiment_id>/
CLI는 audit / prepare / train / evaluate / control / grid / report / all / status / resume을 제공한다. module entry로 실행 가능하게 하고 실제 확인한 Python executable을 사용한다. 필요하면 새 환경을 프로젝트 내부에 만들고 기존 환경을 파괴적으로 변경하지 않는다.
smoke는 소수 run, 1개 horizon, 소수 quantile로 실행 연결·인과성·집계를 확인한다. full 실험 성적을 대신하는 결과로 사용하지 않는다.
본 실험은 launch_full.sh에서 nohup으로 실행할 수 있게 한다. launcher는 고유 experiment_id를 생성하고 해당 ID의 config snapshot, PID, 로그, UTC/KST 시작 시각, git/dirty diff hash, data hash, 환경, stage 상태를 기록한다. PID뿐 아니라 exit code, heartbeat, process identity로 실행 상태를 판정한다.
기존 프로세스를 종료하지 않고 새 실험 자체의 lock으로 중복 실행을 방지한다. CPU/memory를 확인하고 초기에는 최대 4 CPU 수준의 전체 thread budget으로 운영한다. LightGBM/OpenMP와 외부 병렬화가 중첩되어 thread가 과도하게 생성되지 않도록 한다. 학습을 대규모 GPU 작업으로 대체하지 않는다.
checkpoint는 config/data/code fingerprint가 일치하는 task에만 재사용한다. task 단위로 atomic save하고 동일 prediction key의 중복을 방지한다. 실패한 stage의 원인을 기록하고 독립적으로 실행 가능한 stage는 계속 진행한다. 외부 모델 미확보는 blocked_dependency, 미구현은 not_implemented, 실행 중은 running, 실패는 failed로 구분한다.
11. 출력·표·그림
필수 결과:
- inventory.csv, exclusions.csv, capability_matrix.csv, metadata_schema.json
- split_manifest.json, candidate_manifest.json, selected_hyperparameters.csv, requested_resolved_fitted_params.csv
- per_target_predictions.parquet, per_run_metrics.csv, aggregate_metrics.csv
- transition_metrics.csv, tail_metrics.csv, calibration_metrics.csv, paired_gain_ci.csv
- metadata_ablation.csv, control_metrics.csv, grid_metrics.csv
- grid_model_provenance.json, assumptions.csv, verification.json
- status.json, completion_status.json, execution.log, report_ko.md
prediction key는 experiment/dataset/group/run/segment/fold/seed/model/target/horizon/issuance/target-time이다. target start/end, availability, unit, measurement boundary도 저장한다.
5단계로 연결하기 위해 재생성 가능한 make_tables/make_figures를 구현한다. 보고서는 한국어, 그림 label은 영어도 허용한다. 저장된 결과에서 PNG 300 dpi/PDF/figure-data CSV를 생성한다.
- horizon–CRPS, horizon–p95/p99: history vs metadata
- 상승/하강/안정 구간별 오차와 coverage
- nominal quantile–empirical coverage
- 시간–실제 전력/예측/구간: issuance와 target timestamp를 정확하게 정렬
- 제어 전후 PCC power/ramp, GPU command, fast-storage/ESS power, SoC
- 제어 전후 grid frequency/voltage, case별 최대 편차와 회복시간
- metadata gain과 제어/계통 개선의 대응표
실험 결과 없는 그림을 만들지 말고 미생성 이유를 남긴다. simulation/assumed actuator 등 evidence level을 그림과 표에 명시한다. 최악/최선 사례만으로 결론 내리지 말고 전체 case 집계와 사례 자동 선택 규칙을 제공한다.
12. 필수 검증과 완료 보고
tests: 미래 값 perturbation이 과거 feature를 바꾸지 않음, split 간 원본 비중복, target 경계, unknown metadata 처리, gap reset, quantile score, quantile ordering, aggregate 재계산, event matching, storage energy balance, actuator delay, P/Q·SoC 제약, resume 일관성을 확인한다. 실험의 구체적 오류 위험을 검증하고 구현을 복사한 테스트를 늘리지 않는다.
최종 보고서는 다음 다섯 질문에 답한다.
1. Metadata가 새로운 run의 예측을 개선했는가? 평균과 tail/transition 중 어디에서 효과가 있는가?
2. 예측 개선이 같은 controller의 출력·잔여 변동·비용을 개선했는가?
3. 계통 응답을 개선했는가? 어떤 case와 evidence level에서 확인했는가?
4. 개선이 없거나 악화하는 조건은 무엇인가? 독립 run/case 수가 충분한가?
5. 식별되지 않은 actuator/performance/grid 가정에 결론이 얼마나 의존하는가?
실행 후 최신 experiment_id에 대응하는 report만 안내한다. 과거 12-run/36-cell/72.2초 보고서를 이번 결과로 제시하지 않는다. 완료 stage와 실행 중·의존 자료 부족 stage를 구분하고 미완료 항목이 있으면 전체 completed라고 기록하지 않는다.
최종 응답에는 새 폴더, 재현 명령, nohup PID와 로그, 진행/재개 명령, 완료된 실험의 실제 수치, 검증 결과, 미완료 항목을 제시한다. 장시간 실험은 정상 기동과 진행 기록을 확인하고 아직 얻지 않은 결과를 추측하지 않는다.