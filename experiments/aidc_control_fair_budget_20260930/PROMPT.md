AIDC 기존 실험을 이어서, **동일한 저장장치 제약과 에너지 처리량 예산에서 반응형 제어와 예측 제어를 비교하는 독립 실험**을 구현하고 실행하라. 계획만 작성하지 말고, 데이터·코드 확인 → 구현 → smoke test → 가능한 전체 실험 → 결과 분석까지 진행하라.

## 1. 목적과 작업 위치

프로젝트:
`/home/work/AIDC`

기존 실험:
`experiments/aidc_grid_metadata_20260929`

기존 결과:
`experiments/aidc_grid_metadata_20260929/outputs/full_20260928T180455Z`

새 작업 폴더:
`experiments/aidc_control_fair_budget_20260930`

동일 이름의 폴더가 이미 있으면 내용을 확인하고 사용자 변경을 보존하라. 기존 데이터·모델·결과는 덮어쓰지 말고, 필요한 부분을 read-only adapter로 재사용하라. AGENTS.md와 git status를 먼저 확인하라.

핵심 질문은 다음이다.

> 동일한 저장장치 출력·용량·응답 지연·에너지 처리량 예산에서 metadata 예측 제어가 history-only 예측 제어 및 반응형 제어보다 PCC proxy의 ramp와 전력 변동을 줄이는가?

현재 단계는 E1 actuator 가정 기반 제어 모의실험이다. 실제 장치에는 명령을 보내지 말고, 전압·주파수 안정도 개선을 입증했다고 표현하지 마라.

## 2. 기존 구현과 결과 확인

다음 파일과 이를 생성한 코드를 확인하라.

- control_metrics.csv
- per_target_predictions.parquet
- completion_status.json
- controller 설정·구현
- split manifest, 모델 선택 기록, 실행 config

설계상 제어기 정의는 다음과 같지만 실제 코드와 대조하라.

- C0: 무제어
- C1: GPU 계층 제어
- C2: causal reactive hierarchical control
- C3: history-only predictive control
- C4: metadata predictive control
- C5: perfect-future 입력을 사용하는 진단용 제어

기존 분석에서 C4는 C2보다 전력 범위를 조금 줄였지만, fast+ESS 절대 에너지 합계는 약 7.19% 높았다. 이것이 이번 공정 비교의 동기다. 이 수치는 원본에서 다시 계산하고, 집계 방식이 다르면 차이를 설명하라.

특히 아래를 확인하라.

1. C0와 C1이 동일한 원인.
2. C5가 주요 평활화 지표에서 최선이 아닌 원인.
3. 예측 발행 시각·target 시각·제어 적용 시각의 정렬.
4. 초기·종료 시점의 제어 출력과 SoC 처리.
5. 현재 제어기가 어떤 예측값과 horizon을 사용하는지.
6. C3/C4의 예측이 해당 test run을 학습에 사용하지 않은 out-of-fold 결과인지.

구현 오류가 확인되면 새 폴더에서 수정하고, 기존 결과와 수정 후 결과를 구분하라. C5는 최적화된 상한임이 증명되지 않는 한 “perfect-forecast diagnostic”으로 표기하라.

필수 자료가 없으면 기존 프로젝트 내에서 찾아보고, 끝내 확보하지 못한 비교만 blocked로 남겨라. 집계 CSV에서 시계열이나 예측값을 만들어내지 마라.

## 3. 모든 제어기에 공통으로 고정할 조건

각 비교에서 다음 조건을 동일하게 유지하라.

- 평가 run, segment, timestamp와 유효 예측 구간
- fast storage와 ESS 각각의 최대 충·방전 출력
- 각각의 에너지 용량
- 초기 SoC, SoC 상·하한
- 충·방전 효율
- actuator delay와 ramp limit
- 제어 주기와 관측 지연
- 종료 SoC 정책
- GPU 부하와 기타 actuator의 가정

C2/C3/C4 사이에서 달라지는 것은 미래 정보와 train 내부에서 선택한 제어 parameter뿐이어야 한다. C3/C4에는 같은 feature 가용 시점과 동일한 계산·통신 지연 가정을 적용하라.

정보 누출을 막기 위해 test run 전체의 최대값·분위수·향후 에너지를 용량 결정이나 현재 제어 판단에 사용하지 마라.

## 4. 에너지 처리량 예산 정의

각 저장장치의 절대 에너지 처리량을 다음처럼 정의하라.

`E_abs = Σ |P_storage(t)| × Δt / 3600`

전력 단위는 W, 시간은 s, 결과는 Wh이다.

이 값은 충·방전 처리량이다. 순 소비에너지, 효율 손실, 배터리 열화량과 혼동하지 마라. 측정 경계가 AC 측인지 DC 측인지 명시하라.

주 비교에서는 fast storage와 ESS의 예산을 각각 제한하라. 합계만 제한하면 서로 다른 장치로 부담을 옮겨 공정 비교를 왜곡할 수 있다.

예산 기준은 outer-training 자료의 C2 운전에서 산출한 장치별 평균 절대 출력으로 정하고, 예산 배수는 사전에 다음처럼 고정하라.

`0.5, 0.75, 1.0, 1.25, 1.5`

Primary budget은 1.0으로 둔다. Test 결과를 보고 유리한 배수만 선택하지 마라. 기준 처리량이 0이면 임의의 작은 값으로 나누지 말고 별도 처리하라.

평가 길이가 다른 run을 비교할 수 있도록 고정 길이 budget window 또는 온라인 token-bucket 방식으로 예산을 집행하라. 창 길이·초기 credit·최대 credit·종료 처리 규칙은 train에서 고정하고 모든 제어기에 동일하게 적용하라. 알려지지 않은 run 종료 시각을 제어 입력으로 사용하지 마라.

예산 소진 시 feasible action으로 제한하고, 다음을 기록하라.

- 예산 소진·제약 활성 시간 비율
- 요청 출력과 실제 적용 출력의 차이
- 장치별 실제 사용 처리량
- 예산 미사용량

**같은 예산이 실제 사용량까지 같다는 뜻은 아니다.** 같은 예산 비교를 primary로 수행하고, 실제 처리량도 함께 보고하라. 사용량을 억지로 맞추려고 불필요한 충·방전을 추가하지 마라.

## 5. 제어기 tuning과 비교 방법

예측 모델은 우선 기존 모델과 out-of-fold prediction으로 고정하라. 이번 실험에서 모델 재학습과 제어기 변경을 동시에 섞지 마라.

제어기 parameter는 outer-training 내부 grouped validation에서 선택하라. C2/C3/C4에 동일한 후보 수와 탐색 예산을 제공하라.

선택 기준은 test 확인 전에 고정한다.

1. 출력·SoC·처리량 예산·종료 조건 충족.
2. Validation run-balanced p95 absolute PCC ramp 최소화.
3. 동률이면 최대 absolute ramp, 처리량, 단순한 제어기 순으로 선택.

p95 개선과 동시에 최대 ramp가 악화하는지 반드시 별도로 보고하라. 최대 ramp 보호 제약을 추가한다면 그 기준도 train에서 결정하라.

기존 outer split을 재사용하되, 이미 분석한 test run이므로 결과를 “기존 평가셋 재사용 분석”으로 표시하라. 새로운 독립 검증처럼 표현하지 마라.

## 6. 평가 지표와 집계

각 run·segment·controller·budget에 대해 다음을 저장하라.

전력 변동:
- p95/p99 absolute ramp
- maximum absolute ramp
- peak-to-peak power
- 사전 정의된 ramp threshold 초과율과 초과 지속시간
- signed ramp의 상승·하강별 지표

제어 자원:
- fast/ESS 각각의 absolute energy throughput
- 실제 손실 에너지: 효율 모델로 계산 가능한 경우
- SoC 최솟값·최댓값·종료값
- 출력·SoC·예산 제약 활성 비율
- 제어 출력의 최대 변화율

정합성:
- power balance residual
- energy balance residual
- 제어 포화·예산 소진 여부

종료 SoC가 다르면 저장에너지를 더 소모해 얻은 개선일 수 있다. 공통 복구 정책과 고정 평가 구간을 적용하거나, 종료 에너지 차이를 명시해 같은 종료 조건의 결과를 별도로 비교하라. 복구 과정에서 생긴 ramp와 처리량을 숨기지 마라.

Run-balanced 집계를 primary로 사용한다. Segment별 p95를 평균한 값과 run 전체 시계열의 p95를 명확히 구분하라. 연속성이 없는 segment 경계를 연결해 ramp를 만들지 마라.

C4−C3, C4−C2의 paired 차이를 산출하고, 독립 group 단위 cluster bootstrap 2,000회로 95% CI를 계산하라. Segment나 timestamp를 독립 표본으로 취급하지 마라. 고정된 예측·선택 결과의 CI임을 명시하라.

## 7. 필요한 표와 그림

표:
1. 공통 장치 제약과 가정.
2. Budget별 C2/C3/C4 성능·처리량.
3. Primary budget에서 paired 차이와 CI.
4. Run별 개선·악화·동률 수.
5. 제약 위반과 종료 SoC 차이.

그림:
1. 예산 배수에 따른 p95 ramp.
2. 실제 에너지 처리량과 p95 ramp의 trade-off.
3. 실제 처리량과 maximum ramp의 trade-off.
4. C4−C3 및 C4−C2의 run별 paired 차이.
5. 예측 이득이 제어 이득으로 연결된 사례와 연결되지 않은 사례의 시계열.

장치별 처리량을 따로 보여라. 합계 처리량 그림은 보조 분석으로 둔다. 서로 실제 사용량이 다른 점을 “동일 처리량”이라고 표시하지 마라.

사례 선택 규칙을 명시하고, 좋은 사례만 제시하지 마라. 예측·부하·PCC proxy·저장장치 출력·SoC·예산 잔량을 시간 정렬해서 표시하라.

## 8. 구현·실행·산출물

새 폴더에 재현 가능한 config, 실행 entry point, 분석 코드, README를 작성하라. Smoke와 full 실행, resume를 지원하고 config hash가 다른 작업을 잘못 이어 붙이지 마라.

필수 산출물:
- resolved_config.yaml
- provenance.json
- implementation_audit.md
- fair_budget_metrics.csv
- paired_comparisons.csv
- per_run_metrics.csv
- constraint_checks.csv
- completion_status.json
- figures/
- report_ko.md

출력과 SoC 제약, 예산 집행, power/energy balance, timestamp 정렬, 미래 정보 누출에 대해 의미 있는 검증을 수행하라.

Smoke가 통과하면 전체 실험을 실행하라. 계산량이 크면 완료 상태를 저장하고 재개 가능하게 만들되, 승인된 작업을 임의로 smoke에서 끝내지 마라.

보고서는 다음을 직접 답해야 한다.

- 동일 예산에서 C4가 C2/C3보다 나은가?
- 실제 처리량 차이까지 고려해도 이점이 남는가?
- p95 ramp 개선이 최대 ramp 악화를 동반하는가?
- 결과가 특정 run에 집중되는가?
- 예측 정확도 개선이 제어 성능으로 연결되지 않는 원인은 무엇인가?
- 현재 자료로 판정할 수 없는 것은 무엇인가?

Metadata 제어의 우월성을 목표로 결과를 선택하지 마라. 개선이 없거나 악화했다면 그대로 보고하고, 관측된 사실과 원인 가설을 구분하라.

최종 응답에는 실행한 명령, 완료·차단된 범위, 핵심 수치, 산출물 경로와 재개 명령을 제시하라.

## 추가 사용자 요청 원문

### 백그라운드 실행 필수: nohup

전체 실험은 `nohup`을 사용해 백그라운드에서 실행하라. 현재 터미널이나 SSH 연결이 종료돼도 실행이 지속되도록 stdin을 `/dev/null`에 연결하고 stdout·stderr를 실행별 로그에 저장하라.

1. 데이터·코드 확인과 짧은 smoke test를 먼저 수행한다.
2. Smoke가 통과하면 전체 실험 → 결과 집계 → 표·그래프 생성 → 한국어 보고서 작성을 하나의 pipeline으로 연결한다.
3. Pipeline을 `nohup`으로 실제 실행한다. 실행 명령만 제시하고 끝내지 마라.
4. Python 출력은 `-u` 또는 `PYTHONUNBUFFERED=1`로 버퍼링을 해제한다.
5. 실행 디렉터리에 다음을 기록한다.
   - `pipeline.log`
   - `pipeline.pid`
   - `progress.json`
   - `completion_status.json`
   - 최종 exit code
6. 동일 config의 작업이 이미 실행 중이면 PID와 명령행을 확인하고 중복 실행하지 마라. PID 파일만으로 실행 여부를 판단하지 마라.
7. 실행 직후 프로세스 생존과 로그의 초기 진행을 확인한다. 실패하면 원인을 수정한 뒤 재실행하고 이전 로그를 보존한다.

실행 형태는 다음을 따르되, 실제 구현한 경로와 인자로 완성하라.
```bash
nohup bash /absolute/path/run_pipeline.sh \
  --config /absolute/path/config.yaml \
  --output-dir /absolute/path/output \
  > /absolute/path/output/pipeline.log 2>&1 < /dev/null &

pipeline_pid=$!
printf '%s\n' "$pipeline_pid" > /absolute/path/output/pipeline.pid
```

`run_pipeline.sh`는 내부 단계를 순차 실행하고 실패를 상위 단계로 전달해야 한다. 학습·제어 실험이 실패했는데 성공 상태로 보고서를 생성하거나 전체 완료로 표시하지 마라. 기존 결과만으로 생성한 시각화는 별도 완료 항목으로 기록하라.

Checkpoint와 resume를 지원하고, 재개 시 config hash와 입력·모델 identity를 확인하라. `nohup`은 서버 재부팅이나 강제 종료까지 복구해 주지 않으므로, 복구는 checkpoint를 통해 처리하라.

전체 실행이 오래 걸리면 완료될 때까지 대화를 붙잡아 두지 말고, 정상 실행을 확인한 뒤 다음을 보고하라.

- 상태: 실행 중 / 완료 / 실패
- PID와 실행 디렉터리
- 로그 확인 명령
- 진행 상태 확인 명령
- 해당 작업만 안전하게 중지하는 방법
- 중단 후 재개 명령
- 완료 후 보고서·표·그래프가 저장될 경로

백그라운드 시작을 실험 완료로 표현하지 마라. 최종 결과 분석은 실제 완료된 산출물을 기준으로 작성하라.

프롬프트 자체도 md로 남기도록

## 후속 지시

계속 진행
