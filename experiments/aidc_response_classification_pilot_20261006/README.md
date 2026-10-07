# AIDC 대응 수준 분류 pilot

실측 GPU 8개 합산 전력으로 고정 PCC proxy 요구조건의 대응 수준을 예측한다. 원본 전처리·feature·sampling 구현과 기존 feasibility 판정기를 재사용했다. 기존 회귀 실험과 `outputs/full_v1`은 보존했다.

`configs/pilot.yaml`은 **가상 공학 pilot 가정**이다. GPU sum을 PCC proxy로 사용하고 ramp 500 W/s, 관측 복구 1 s, 기본 100 W·7,200 J 및 추가 300 W·21,600 J storage를 적용한다. 자원은 실측/제조사 검증 장치가 아니다. 원래 미확정 `response_label_spec.yaml`도 보존하며, 해당 설정을 선택하면 필수값 gate가 계속 작동한다. 설정을 결과에 맞춰 조정하지 않는다.

흐름: 원본 CSV hash·bin 재구성 확인 → 연속 history와 관측 미래 확보 → label과 독립적으로 anchor sampling → P_base(최신 완료 bin W)와 U0/U1/U2 hard MILP·물리 재검산 → R0~R3/U → 개발 날짜별 M1/M2 학습 → baseline 비교·run-balanced 평가 → 한국어 보고서·그래프.

개발 날짜는 2025-09-16/17/20, 보류는 2025-09-22다. 날짜·run·sample key의 train/validation 겹침을 거부한다. 모델 feature는 현재·과거 값과 알려진 metadata만 사용한다. 미래 actual은 oracle label 생성에만 사용한다. U는 학습에서 제외하고 infeasible과 solver/data 실패를 구분한다. run당 학습/평가는 최대 30/60 anchor, 2개 모델×3개 개발 folds+1개 final=8 configurations다. 기존 LightGBM 설정을 유지하고 추가 모델·의존성·탐색을 도입하지 않았다.

## 실행·로그·중지·재개

`/home/work/AIDC`에서 실행한다. 아래 새 출력 경로는 기존에 없는 경로로 지정한다.

```bash
exp_dir=experiments/aidc_response_classification_pilot_20261006
spec_path="$exp_dir/configs/pilot.yaml"
out_dir="$exp_dir/outputs/my_pilot"
python -m unittest discover -s "$exp_dir/tests" -p 'test_*.py' -v
python -u "$exp_dir/pipeline.py" --mode preflight --spec "$spec_path" --output "$out_dir" > /tmp/aidc_preflight.log 2>&1
# preflight가 exit 0이고 label 검증·classifier smoke가 통과한 다음 실행한다.
python "$exp_dir/manage.py" start --spec "$spec_path" --output "$out_dir"
tail -f "$out_dir/pipeline.log"
python "$exp_dir/manage.py" status --output "$out_dir"
python "$exp_dir/manage.py" stop --output "$out_dir"
python "$exp_dir/manage.py" resume --spec "$spec_path" --output "$out_dir"
```

`manage.py start/resume`은 `nohup python -u`·stdin `/dev/null`·독립 process group으로 실행하며 PID/start ticks/명령을 저장한다. 완료는 `completion_status.json: all_completed=true`와 `exit_status.json: exit_code=0`으로 확인한다. 중지 요청은 영속 `stop_requested.json`과 SIGTERM으로 전달한다. native solver가 signal 처리를 지연할 때도 다음 단계·fit 경계에서 요청을 확인한다. interruption exit는 130이다. PID가 재사용되거나 해당 실행이 이미 종료됐으면 signal을 보내지 않는다. lock은 중복 실행을 차단한다.

preflight가 중단된 경우 같은 config/output에 `--resume`을 붙인다. 학습 중단 후에는 위 관리 명령을 사용한다. 입력·코드·설정 identity가 같고 checkpoint checksum이 맞아야만 재개한다. 코드/설정 변경 후에는 새 output이 필요하다. label은 100행 단위, 학습은 fold/model 단위로 checkpoint한다. 상태 JSON·parquet·모델은 임시 파일 후 rename한다. 정상 run 제외는 `excluded_runs.csv`, 판정 실패는 label status/unknown_reason, 실행 실패는 stage를 포함한 상태·로그에 남긴다.

```bash
python -u "$exp_dir/pipeline.py" --mode preflight --spec "$spec_path" --output "$out_dir" --resume
# 완료된 결과에서 학습 없이 보고서 재생성
python -u "$exp_dir/pipeline.py" --mode report-only --spec "$spec_path" --output "$out_dir"
```

## 주요 파일

| 파일 | 역할 |
|---|---|
| `config.py`, `configs/pilot.yaml` | 고정 설정 로드·검증, 가정·단위·출처·class mapping |
| `data.py` | 기존 prepared data 읽기, 유효 anchor·sampling·날짜 split |
| `features.py` | 연속 원본 bin의 인과 feature, 기존 metadata encoding 재사용 |
| `labeling.py` | P_base W, storage hard MILP, 물리 W/J/SoC 재검산, R0~R3/U |
| `training.py` | baseline, 고정 M1/M2 학습·예측, fold checkpoint |
| `evaluation.py` | 수치 지표와 저장 예측 집계 |
| `plotting.py` | 저장 표·예측의 PNG/SVG/CSV 그래프 |
| `reporting.py` | 한국어 Markdown/HTML 보고서 |
| `pipeline.py`, `manage.py` | 단계 실행·로그·checkpoint·resume·nohup·안전한 중지 |
| `source_audit.py`, `source_evidence.csv` | 원본 감사와 기존 출처·가상 pilot 채택 기록 |
| `common.py` | 기존 hash·원자 저장·feature helper 로드 |
| `tests/test_labeling.py` | label 예시, solver 실패, feature 인과성, split, 단위, train-only encoding |

`labels.py/prepare.py/modeling.py/test_pilot.py`는 이전 import/테스트 명령 호환용이다. 구현을 중복하지 않는다. import만으로 학습·실험 산출물 생성은 실행하지 않는다. Ruff는 설치돼 있지 않아 설치된 Black으로 이 실험의 수정 Python 파일만 정리하고 Pyflakes로 검사했다.

## 결과와 해석

이번 최종 실행 경로와 검증 수치는 아래 완료 기록을 따른다. 각 output의 `report_ko.md`, `report_ko.html`, `figures/`에서 보고서를 확인한다. `labels.parquet`, `predictions.parquet`, `aggregate_metrics.csv`, `per_run_metrics.csv`, `class_metrics.csv`, `metadata_effect.csv`는 label·예측·평가 결과다. `resolved_config.yaml`, `feature_schema.json`, `fit_registry.json`에 실제 설정·feature 목록·class 좌표·범주 mapping을 저장한다.

주장할 수 있는 것은 이 고정 가상 자원/proxy 설정에서의 실행 재현성, oracle label 분류 성능과 baseline/metadata 비교다. 실제 PCC 제어 가능성, 장치 구현 가능성, 주파수·전압 안정성, 계통 붕괴 여부, 독립 외부 일반화는 주장할 수 없다. 보류 날짜는 과거 실험에서도 사용했고 session 독립성은 미확인이다. 구간은 겹치므로 anchor 개수는 독립 표본 수가 아니다.

## 이번 완료 기록 — 2026-10-07 KST

최종 결과: [`outputs/quality_pilot_final_v2/report_ko.md`](outputs/quality_pilot_final_v2/report_ko.md), [HTML](outputs/quality_pilot_final_v2/report_ko.html), [독립 대조 검증](outputs/quality_pilot_final_v2/final_verification.json). `all_completed=true`, exit 0, 본 fit 8회·smoke fit 1회. 원본 32개 감사 및 핵심 테스트 16개 통과. 고유 label anchor 2,535개(R0 922·R1 442·R2 123·R3 1,048), U 0·nesting 모순 0.

실제 stop은 exit 130으로 기록됐고 resume은 `fit_0_M1.pkl`, `fit_0_M2.pkl`의 mtime을 유지했다. 별도 report-only도 신규 fit 없이 완료됐다. label·예측이 중지 지원 보완 전의 동일 설정 실행과 일치했다. 비교에 사용한 임시 실행 디렉터리는 검증 후 정리했다. 기존 `outputs/full_v1` 및 회귀 결과는 보존했다.

Run-balanced ordinal MAE는 개발 M1/M2=1.152/1.196, 보류=1.109/0.622이며 Always-R0 baseline은 개발/보류=1.338/2.098이다. Metadata 효과는 개발에서 악화·보류에서 개선으로 일관적이지 않다. 이 숫자는 고정 가상 자원 설정에 조건부이며 실제 계통·장치 검증 결과가 아니다.
