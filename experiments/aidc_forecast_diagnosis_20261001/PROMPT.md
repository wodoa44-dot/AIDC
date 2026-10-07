현재 AIDC 예측·제어 실험을 이어서, **예측모델 자체의 성능 한계와 예측을 제어에 연결하는 방식의 한계를 분리해 판정하는 최종 진단 실험**을 구현하고 실행하라.

계획만 제시하지 말고 기존 결과 감사 → 진단 실험 → 필요한 제한적 재학습 → 제어 비교 → 표·꺾은선 그래프 → 한국어 보고서까지 수행하라.

목표는 metadata나 예측 제어의 우월성을 만드는 것이 아니다. **현재 데이터와 평가 조건에서 어떤 예측이 유효하고, 무엇이 부족하며, 추가 모델 개발을 계속할 근거가 있는지 확인하는 것**이다.

## 1. 작업 위치와 기존 결과 보존

프로젝트:
`/home/work/AIDC`

기존 예측 실험:
`experiments/aidc_grid_metadata_20260929`

기존 공정 비교 실험:
`experiments/aidc_control_fair_budget_20260930`

최근 nested 결과:
`experiments/aidc_control_fair_budget_20260930/nested/outputs/full_v2`

새 작업 폴더:
`experiments/aidc_forecast_diagnosis_20261001`

AGENTS.md와 git status를 확인하라. 기존 코드·결과·사용자 변경은 보존한다. 기존 모델과 OOF prediction은 provenance가 맞으면 재사용하고, 새 산출물은 별도 실행 디렉터리에 기록한다.

실제 GPU·UPS·ESS에는 명령을 보내지 마라. 제어 평가는 E1 actuator 가정 기반 PCC proxy 실험이다.

## 2. 현재 관측 결과와 확인할 질문

최근 보고서에서 primary budget 1.0의 결과는 다음과 같다. 원본 CSV에서 다시 확인하라.

- C2 p95 ramp: 약 2508.5 W/s
- C3 p95 ramp: 약 2640.1 W/s
- C4 p95 ramp: 약 2686.9 W/s
- C4는 C2보다 저장장치 처리량도 많음
- Predictor는 1초 endpoint, history 1초
- 15 quantiles를 학습했지만 제어에는 q50만 사용
- Forecast weight는 0.5 고정
- 원본 32개 run 중 실제 제어 평가 23개
- 60초 창 중 마지막 20초는 SoC 복구
- 독립 수집 group 미확정, 날짜 4개로 가정한 leave-day-out 평가

다음 질문을 각각 분리해 답하라.

A. 같은 최신 split에서 P4는 P2보다 실제 예측이 좋은가?
B. 예측오차를 제거해도 현재 제어기는 더 좋아지는가?
C. 예측 반영 비중을 선택하면 성능이 개선되는가?
D. 1초 endpoint라는 target이 제어에 필요한 정보를 충분히 제공하는가?
E. 악화가 실제 운전 구간과 복구 구간 중 어디서 발생하는가?
F. 현재 데이터에서 추가 예측모델 개발을 계속할 근거가 있는가?

## 3. 먼저 평가·구현을 감사하라

### 3.1 Run 제외 감사

32개 run 모두에 대해 포함·제외 manifest를 작성한다.

- 원본 길이와 연속 segment 길이
- 사용 가능한 feature/target 표본 수
- 유효 제어 window 수
- 제외 사유
- Hardware·workload·model·batch·수집일

23개에 포함된 run과 제외된 9개 run의 특성을 비교한다.

제어 window가 없다는 이유만으로 예측 평가에서도 제외하지 마라. 예측 평가 가능한 전체 run과 제어 비교 공통 run의 결과를 각각 보고한다.

제외 규칙을 test 성능에 맞춰 바꾸지 마라.

### 3.2 시간 정렬과 인과성

관측 bin, 예측 발행 시각, target bin의 시작·끝, 명령 시각, actuator 적용 시각을 명시한다.

미래 값을 feature·encoding·budget calibration·controller tuning에 사용하는 경로가 없는지 검사한다.

Ramp는 실제 Δt로 계산하며 단절된 segment/window 사이를 연결하지 마라. Endpoint 변화량을 구간 최대 ramp로 취급하지 마라.

### 3.3 Controller 수식과 구현

현재 controller를 수식과 의사코드로 정리한다.

- 예측값이 reference에 들어가는 방식
- Forecast weight의 정확한 의미
- Fast/ESS 출력 분담
- Delay·slew·포화 처리
- Budget 제한과 예약
- SoC 복구와 종료 출력 처리

Forecast weight=0에서 같은 parameter의 반응형 제어와 출력·상태가 일치하는지 검증한다. 일치하지 않으면 이유를 규명하고, 공통 controller core로 비교 가능한 진단 경로를 구현한다.

SoC·출력을 종료 시점에 강제로 덮어쓰는 방식은 금지한다. 종료 조건은 물리 제약과 에너지 수지를 만족하는 실제 제어로 달성해야 한다.

## 4. 단계 A: 최신 split에서 예측모델 자체 평가

기존 수정된 outer split과 누출 없는 OOF prediction을 우선 사용한다.

평가 모델:
- Persistence: 현재 완료 bin의 전력을 미래 예측으로 사용
- 기존 history-only P2
- 기존 metadata P4

계절성·주기 기반 baseline은 training에서 반복 주기가 확인될 때만 추가한다.

과거 배포 모델을 비교하려면 test run을 학습에 사용했는지 확인한다. 학습 이력이 겹치면 독립적인 일반화 비교로 포함하지 마라.

평가 항목:
- MAE, RMSE, signed bias
- p95/p99 absolute error
- Quantile pinball loss와 기존 정의의 CRPS approximation
- q10–q90 coverage와 interval width
- 상승·하강·정상 구간별 오차
- 큰 전이 구간의 과소·과대 예측
- Run·날짜·hardware·workload별 결과

Persistence 점예측과 quantile 모델의 점예측은 같은 point metric으로 비교한다. Persistence를 퇴화분포로 평가하면 그 사실을 표시하고, 서로 다른 CRPS approximation을 혼용하지 마라.

이벤트 threshold는 기존 사전 정의를 재사용하거나 training에서만 결정한다. Test 예측오차가 큰 지점을 보고 이벤트 정의를 변경하지 마라.

보고서에 반드시 다음 두 표를 포함한다.
1. 예측 평가 가능한 전체 run.
2. 제어 평가와 공통인 run·timestamp.

이전 5-fold 결과의 개선율을 이번 leave-day-out 결과에 가져오지 마라.

## 5. 단계 B: 완벽한 예측을 넣어 원인을 분리

모델을 더 학습하기 전에, 동일한 장치·예산·초기·종료 조건에서 다음 입력을 비교하라.

- Persistence endpoint
- P2 endpoint q50
- P4 endpoint q50
- Perfect endpoint: 실제 1초 후 target 값

Perfect endpoint는 진단용 미래정보이며 배포 가능한 controller가 아니다. 이를 최적 제어 성능의 상한이라고 부르지 마라.

두 비교를 분리한다.

### B1. 정책 고정 입력 교체

같은 controller parameter를 고정하고 예측 입력만 바꾼다. 예측오차를 제거했을 때의 효과를 확인한다.

### B2. 동일 절차의 내부 tuning

각 입력에 대해 같은 inner validation·목적함수·탐색 예산으로 parameter를 선택한다. Outer-test 성능으로 선택하지 마라.

Perfect endpoint에서도 반응형보다 나아지지 않는다면, endpoint 정보를 현재 controller가 유용하게 활용한다는 근거가 부족한 것이다. 이 결과만으로 예측 전체가 불필요하다고 결론 내리지 마라.

## 6. 단계 C: 예측 반영 비중 검증

Forecast weight 후보:
`0, 0.25, 0.5, 0.75, 1.0`

기존 reference slew와 slow alpha 후보를 사용한다. 기존 controller weight의 유효 범위가 다르면 수식으로 대응을 정의하고 실행 전에 config에 고정하라.

다음 두 결과를 구분한다.

1. 다른 parameter를 고정한 weight sensitivity.
2. Inner grouped validation에서 weight와 controller parameter를 함께 선택한 결과.

Weight=0인 동일 행동의 후보는 중복 평가하지 마라. 각 모델에 같은 고유 후보 집합과 선택 규칙을 적용한다. 탐색 예산과 실제 평가 후보 수를 보고하라.

반응형 후보를 포함해도 outer-test에서 반드시 반응형보다 좋아지는 것은 아니다. Inner 선택 오류와 일반화 실패를 그대로 기록하라.

Outer fold·budget별 선택 weight와 validation score를 저장한다. Weight=0의 선택 비율도 결과로 보고한다.

## 7. 단계 D: Endpoint와 미래 궤적의 정보 차이

단계 B/C만으로 원인을 분리하지 못하거나 endpoint의 정보 한계가 의심되면 다음 제한된 확장을 수행하라.

미래 target:
`0.2, 0.4, 0.6, 0.8, 1.0초`에 끝나는 200 ms 평균 전력.

동일한 인과적 history·P2/P4 feature 정의·고정 LightGBM 설정으로 horizon별 q50 모델을 학습한다. 제어 비교에 q50만 필요한 이 단계에서는 불필요한 quantile 모델을 추가 학습하지 마라.

Inner OOF와 outer prediction을 분리하고, 학습 cache identity를 검사한다. 기존 1초 prediction이 호환되면 재사용한다.

궤적을 사용하는 제한된 receding-horizon controller를 별도 실험으로 구현한다. 지원되는 solver를 확인하고 다음을 명시하라.

- 목적함수와 가중치
- 장치 출력·에너지·SoC·delay·slew·처리량 제약
- 기존 복구·종료 정책과의 연결
- Solver 실패 시 인과적 fallback
- 계산시간과 제약 위반

새 controller 내부에서 입력을 다음과 같이 비교한다.

- Persistence trajectory
- P2 predicted trajectory
- P4 predicted trajectory
- Perfect trajectory

**Controller 구조 변경과 예측 정보 변경을 분리하라.** 새 controller의 learned trajectory 결과를 기존 C2하고만 비교해 모든 이득을 예측 효과로 부르지 마라. 같은 새 controller의 persistence 입력을 필수 비교군으로 포함한다.

Perfect trajectory도 해당 controller에서의 진단 결과다. 수학적으로 입증되지 않은 최적 성능 상한이라고 표현하지 마라.

독립 시점 quantile을 연결한 선을 joint trajectory confidence band로 취급하지 마라.

필요한 solver나 데이터가 없으면 이 단계만 구체적인 사유로 blocked 처리하고 A/B/C 결과는 완성하라.

## 8. 운전과 복구 구간을 분리하라

기존 60초 전체 평가를 primary로 유지하면서 다음을 따로 산출한다.

- 운전 구간: 처음 40초
- 복구 구간: 마지막 20초
- 전체 60초

운전→복구 경계의 ramp를 빠뜨리지 않도록 경계 배정 규칙을 고정한다.

각 구간에서:
- p95/p99/max absolute ramp
- Ramp threshold 초과율·지속시간
- 장치별 처리량
- 포화·예산 소진 비율
- Requested/applied 출력 차이

전체 최대 ramp가 발생한 시각과 당시 원인 후보를 기록한다.

- 부하 전이
- Controller reference 변경
- Actuator 포화·해제
- Budget 제한
- SoC 복구
- 종료 출력 복귀

시간적 동시 발생을 원인의 인과적 증명으로 표현하지 마라. 전체 p95를 구간별 p95의 가중합으로 재구성하지 마라.

## 9. 비교 조건과 통계

기존 장치·예산·효율·delay 조건을 primary로 유지한다.

예산 배수:
`0.5, 0.75, 1.0, 1.25, 1.5`

Primary budget:
`1.0`

추가 실험도 같은 표본·종료 조건에서 비교한다. Fold별 장치 calibration이 다른 점을 표기하고, 같은 fold 내 paired 차이를 사용한다.

확인된 독립 수집 group이 없다는 기존 한계를 유지한다. 날짜 group 기반 CI는 가정 기반으로 명시하고, run/날짜별 차이와 leave-one-group-out 민감도를 제공한다.

전체 결과는 이미 확인한 평가셋을 재사용하는 탐색적 진단이다. 새로운 독립 검증이나 확인적 유의성 검정으로 표현하지 마라.

## 10. 실험 확대와 중단 기준

결과를 보고 좋은 조합이 나올 때까지 무제한 탐색하지 마라. A/B/C는 필수로 수행하고 D의 실행 여부와 근거를 기록한다. 추가 모델 구조 탐색은 이번 범위에 포함하지 않는다.

최종 판정은 다음 기준으로 작성한다.

- P4가 최신 split에서 P2보다 예측이 좋은가?
- Perfect endpoint와 learned endpoint 사이에 활용 가능한 성능 차이가 있는가?
- Weight 선택으로 예측 사용의 가치가 나타나는가?
- 궤적 입력이 endpoint의 한계를 줄이는가?
- 개선이 전체 60초에서도 남는가?
- 개선에 더 많은 처리량·최대 ramp 악화가 동반되는가?
- Metadata의 이점이 특정 run·날짜에 집중되는가?

다음 유형으로 결론을 구분하되, 증거가 부족하면 판정 불충분으로 남겨라.

1. 예측오차가 주요 개선 대상이라는 근거가 있음.
2. 현재 controller의 예측 활용 방식이 주요 개선 대상이라는 근거가 있음.
3. Endpoint target보다 궤적 정보가 유용하다는 근거가 있음.
4. 현재 데이터에서 metadata의 추가 가치를 확인하지 못함.
5. 현재 장치·예산 조건에서 반응형 대비 예측의 추가 가치를 확인하지 못함.
6. 데이터 부족·교락·구현 제약으로 구분 불가.

하나의 실험으로 모든 가능한 예측모델의 한계나 예측 불가능성을 증명했다고 주장하지 마라.

## 11. 표와 꺾은선 그래프

필수 표:
- 32개 run 포함·제외 사유
- 최신 split의 예측 성능
- 동일 정책에서 예측 입력 교체 결과
- Weight tuning 전후 결과와 선택 weight
- Endpoint/trajectory 비교
- 운전·복구·전체 구간별 성능
- 장치 처리량·종료 조건·제약 위반
- 최종 판정과 근거·한계

필수 그래프:
1. Forecast weight에 따른 p95/max ramp.
2. Budget에 따른 제어 성능.
3. 실제 처리량과 제어 성능의 trade-off.
4. Horizon별 P2/P4/persistence 예측오차.
5. Run별 예측 개선량과 제어 개선량의 scatter.
6. 실제 부하·예측·PCC·장치 출력·SoC·예산 잔량의 시간 꺾은선.
7. 운전·복구 구간과 최대 ramp 발생 시각을 표시한 확대 그림.

Ordinal/continuous X축에만 꺾은선을 사용한다. 순서 없는 모델명을 임의의 연속축처럼 연결하지 마라.

성공·실패·중간 사례를 사전 정의한 규칙으로 선택한다. Run 전체의 평가 결과와 실제 표시한 window의 결과를 함께 기록하여, 첫 window가 run 전체 현상을 대표한다고 가정하지 마라.

PNG 300 dpi와 SVG, plotting CSV를 저장한다. 표·그림에 단위·집계 방식·표본 수·가정 기반 CI·E1 조건을 명시한다.

Markdown과 HTML 한국어 보고서를 생성하고 실제 렌더링에서 글자·축·범례 잘림을 확인한다.

## 12. nohup 실행과 재개

짧은 smoke test 후 전체 pipeline을 **nohup으로 백그라운드 실행**하라.

Pipeline:
감사 → 예측 평가 → 입력 교체 → weight 실험 → 조건부 궤적 실험 → 집계 → 시각화 → 보고서.

- Python은 unbuffered 출력.
- stdin은 /dev/null.
- stdout/stderr는 실행별 pipeline.log.
- PID·시작 시각·명령행·config hash 기록.
- progress.json·completion_status.json을 원자적으로 갱신.
- 단계별 checkpoint와 resume 지원.
- 동일 작업 중복 실행 방지.
- 실패 exit code와 실패 단계를 보존.
- 기존 결과에서 시각화만 재생성하는 모드 제공.

독립 group 자료가 없다는 이유로 실행 가능한 계산 전체를 중단하지 마라. 반대로 독립 group 검증을 완료한 것처럼 표시하지 마라.

정상 시작과 초기 로그를 확인한 뒤 실행 상태·PID·로그 확인·중지·재개 명령을 제공한다. 백그라운드 시작을 완료로 표현하지 마라.

최종 보고서는 **무엇을 더 학습해야 하는지, 제어를 바꿔야 하는지, 현재 데이터로는 여기서 멈추는 것이 타당한지**를 실제 결과에 근거해 명확히 답하라.