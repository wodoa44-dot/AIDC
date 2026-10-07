# 후속 사용자 요청 원문

현재 고정 정책의 부분 결과를 보존하고, 미완료한 다음 요구를 이어서 수행하라.

1. Outer-training 내부 grouped validation을 통한 C2/C3/C4 parameter 선택과 비교.
2. 독립 수집 group 확인에 기반한 통계적 검증.

기존 시각화·보고서·nohup 실행 요구는 유지한다. 처음부터 전체 작업을 다시 만들지 말고 기존 구현·checkpoint를 재사용하라.

### 1. 재학습 조건 수정

기존 “예측 모델 재학습 금지” 조건을 다음처럼 수정한다.

**내부 검증용 out-of-fold prediction 생성에 필요한 동일 모델 계열·동일 feature 정의의 재학습을 허용한다.**

이번 작업에서 새로운 모델 구조·metadata·target·loss를 탐색하지 마라. 재학습은 controller tuning에 필요한 누출 없는 예측을 확보하는 데 한정한다.

저장된 모델이나 예측이 있으면 provenance와 학습 표본을 확인한 뒤 먼저 재사용한다. 없으면 필요한 fold만 학습하고 캐시하라.

### 2. Nested grouped validation

각 outer fold에서 다음을 수행하라.

1. Outer-test group을 완전히 분리한다.
2. Outer-training group만으로 inner grouped folds를 만든다.
3. 각 inner fold에서 inner-training으로 predictor를 학습한다.
4. 해당 predictor로 inner-validation의 예측을 생성한다.
5. 동일 inner-validation 구간에서 C2/C3/C4의 후보 parameter를 평가한다.
6. Inner-validation 결과만으로 controller parameter를 선택한다.
7. 선택을 고정한 후 outer-test에서 한 번 평가한다.

Inner-validation run을 학습에 포함한 predictor의 예측으로 controller를 tuning하지 마라.

Predictor의 hyperparameter·history·전처리 설정은 사전 고정된 값을 우선 사용한다. 기존 값이 해당 inner-validation label을 이용해 선택된 값이면, 그 값을 누출 없는 설정으로 취급하지 마라. 필요하면 inner-training 내부에서만 선택하고 추가 선택 절차를 기록하라.

결측값 처리·encoding·normalization·예산 기준·제어 threshold 등 학습되는 값도 해당 training partition에서만 결정한다.

C2는 예측 모델이 필요 없지만 C3/C4와 동일한 validation 구간·장치 제약·목적함수·후보 평가 예산으로 tuning하라.

### 3. 독립 수집 group 확인

원본 README, metadata, 수집 로그, session identifier와 timestamp를 확인해 `group_manifest.csv`를 작성하라.

포함 항목:

- run_id
- source_file
- collection_session_id
- independent_group_id
- group 판단 근거
- 확인 상태와 남은 불확실성

동일 수집 session에서 잘라낸 파일·segment는 같은 group으로 묶는다. 중복 또는 겹치는 구간도 독립 표본으로 취급하지 마라.

Hardware·모델·batch가 같다는 이유만으로 동일 수집 group으로 묶지 마라. 반대로 파일명이 다르다는 이유만으로 독립이라고 판단하지 마라.

기존 outer split에서 동일 수집 group이 train/test에 걸쳐 있으면 해당 split을 수정하고 영향받은 예측·평가를 재생성하라. 수정 전 결과는 별도로 보존한다.

독립성을 확인할 자료가 부족하면 임의로 확정하지 마라. 근거가 있는 범위에서 보수적인 grouping과 민감도 분석을 수행하고, 가정에 의존한 결과로 표시한다.

### 4. 통계적 검증

C4−C3, C4−C2를 동일 outer-test 표본에서 paired 비교하라.

독립성이 확인된 group을 단위로 cluster bootstrap을 수행한다. Timestamp·segment·seed를 독립 표본처럼 부풀리지 마라.

Run-balanced 점추정과 독립 group 수를 함께 보고하고, bootstrap에서도 같은 estimand를 유지하라. Group-balanced 결과를 추가하면 별도 이름으로 구분하라.

독립 group 정보가 불충분하면 정식 독립-group CI는 미완료로 남긴다. 가정 기반 CI를 제공할 경우 정식 검증과 분리한다. Group 수가 적을 때는 개별 group 차이와 leave-one-group-out 민감도도 제시하라.

이미 검토한 outer-test를 재사용하므로 “기존 평가셋 재사용 분석”이라는 표기를 유지하라.

### 5. 기존 결과와 새 결과 구분

다음 결과를 섞지 마라.

- fixed_policy: 기존 고정 정책의 부분 결과
- inner_tuned: 내부 grouped validation으로 선택한 정책
- grouping_sensitivity: 수집 group 가정에 따른 민감도 분석

동일 예산·용량·장치 제약에서 비교하고, 기존에 요구한 처리량·종료 SoC·최대 ramp 평가를 유지한다.

표와 꺾은선 그래프에 fixed_policy와 inner_tuned를 구분해 표시한다. Controller tuning 전후의 성능·처리량 변화를 추가한다. 각 그림에 실제 평가 group 수와 CI의 근거를 표기한다.

### 6. 실행과 완료 조건

짧은 smoke test 후 전체 학습·tuning·평가·시각화·보고서 pipeline을 **nohup으로 백그라운드 실행**하라.

로그·PID·진행 상태·checkpoint·exit code를 기록하고, 정상 시작을 확인한 뒤 모니터링·중지·재개 명령을 제공하라. 기존 실행 중인 작업과 중복 실행하지 마라.

`completion_status.json`에서 다음 상태를 별도로 관리하라.

- inner_oof_predictions
- controller_grouped_tuning
- collection_group_verification
- outer_evaluation
- independent_group_statistics
- visualization
- report

필수 항목이 실제 완료된 경우에만 `all_completed: true`로 바꿔라. 독립 group 확인이 막혀도 가능한 tuning·평가·시각화는 계속 수행하고, 통계적 결론의 제한만 명확히 남겨라.

최종 보고서에는 재학습한 범위, 누출 방지 방법, group 판단 근거, 고정 정책 대비 변화, 남은 미완료 항목을 기록하라.

---
원래 프롬프트와 nohup 실행 요구 원문: [../PROMPT.md](../PROMPT.md)
