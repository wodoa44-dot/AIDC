# 기존 fair-budget 실험의 nested 확장

원래 `../outputs/full` 고정 정책 산출물과 기존 Python 파일은 수정하지 않는다.
물리 simulator, 창 분리, calibration, 집계, nohup 관리 코드를 재사용한다.

## 분할과 학습

원본 README, 42개 경로의 SHA256, 32개 고유 원본 CSV timestamp를 확인한다.
파일 별칭은 통합한다. 명시 session/reset/randomization 로그가 없어 독립 group은 미확인이다.
수집일 4개를 보수적 가정 group으로 삼아 leave-day-out outer 4개, 각 outer의
나머지 수집일 3개를 inner validation으로 순회한다. 기존 run split은 날짜가 양쪽에
걸치므로 수정하고 모든 영향받은 예측을 재생성한다. 날짜와 hardware/workload는 교락되어 있다.

기존 candidate 0의 사전 정의 hyperparameter를 고정한다. P2/P4, 동일 feature,
1초 endpoint target, quantile loss, seed 17, 15 quantiles 및 정렬 후 median을 유지한다.
Full은 12 inner + 4 outer partition × 2 models × 15 quantiles = 480 fits 범위다.
새 모델 구조/feature/loss 탐색 없이 누출 없는 tuning 및 수정 split 평가에만 재학습을 사용한다.
각 partition의 학습 표본·validation 표본·encoding feature hash·모델 checksum을 저장한다.

C2/C3/C4 각각 6개 controller 후보, 같은 5개 예산과 validation 창을 평가한다.
Calibration은 해당 inner-training에서만 수행한다. 기본 p95 → 최대 ramp → 실제 처리량 →
고정 후보 순서로 선택한다. 모든 물리 제약 및 종료 SoC 검증에 실패하면 pipeline 실패다.
선택 파일을 고정한 후 outer predictor를 생성하고 test 평가를 진행한다.

## 산출물 구분

- 원래 `../outputs/full`: legacy fixed_policy 그대로 보존.
- 새 output의 `fixed_policy/`: 같은 수정 split/새 predictor/장치/예산에서 고정 정책 재평가.
- `inner_tuned/`: inner-validation으로만 선택한 정책.
- `grouping_sensitivity.csv`: 같은 새 outer 평가에서 source/date/campaign 가정에 따른 통계 민감도.

Matched fixed-policy 대비 변화만 tuning 효과로 해석한다. Legacy와의 전체 차이를 tuning 효과라고 부르지 않는다.
Run-balanced 점추정과 cluster sums / resampled run counts로 같은 estimand의 2,000회 bootstrap을 계산한다.
수집일 독립성도 미확인이다. 정식 independent-group CI는 blocked로 유지하며 가정 CI, 개별 group,
leave-one-group-out 민감도를 따로 보고한다. 하나의 campaign 가정은 group 1개라 CI를 만들지 않는다.

## 실행

```bash
cd /home/work/AIDC/experiments/aidc_control_fair_budget_20260930/nested
/usr/bin/python -m unittest -v test_nested.py
/usr/bin/python manage.py launch --config "$PWD/config.yaml" --output-dir "$PWD/outputs/smoke_v2" --smoke
/usr/bin/python manage.py launch --config "$PWD/config.yaml" --output-dir "$PWD/outputs/full_v2"
tail -f outputs/full_v2/pipeline.log
cat outputs/full_v2/progress.json outputs/full_v2/completion_status.json
/usr/bin/python manage.py status --config "$PWD/config.yaml" --output-dir "$PWD/outputs/full_v2"
/usr/bin/python manage.py stop --config "$PWD/config.yaml" --output-dir "$PWD/outputs/full_v2"
/usr/bin/python manage.py launch --config "$PWD/config.yaml" --output-dir "$PWD/outputs/full_v2" --resume
```

`launch`는 실제 `nohup bash run_pipeline.sh ...`를 stdin=/dev/null, stdout/stderr=pipeline.log,
새 session, Python -u로 실행한다. PID·명령행·start ticks 확인 및 config hash 잠금으로 중복을 차단한다.
중지는 검증한 해당 PID에만 pidfd TERM을 보낸다. 원래 작업 결과에는 쓰지 않는다.
재개는 config·코드·원본·모델 identity가 같아야 하며 quantile, 후보, run/controller/budget checkpoint를 재사용한다.
이전 로그·상태는 `.previous`로 보존한다. SIGKILL·재부팅 후에는 실제 생존 여부를 확인해 재개한다.

필수 상태를 `completion_status.json`에서 독립 관리한다:
inner_oof_predictions, controller_grouped_tuning, collection_group_verification,
outer_evaluation, independent_group_statistics, visualization, report.
독립성 검증이 막혀도 가능한 계산은 계속하며 all_completed를 true로 바꾸지 않는다.

Full 최종 파일: `report_ko.md`, `group_manifest.csv`, `group_audit_ko.md`, `provenance.json`,
`split_manifest.json`, `reuse_audit.json`, `selected_parameters.csv`, `validation_scores.csv`,
`fair_budget_metrics.csv`, `per_run_metrics.csv`, `constraint_checks.csv`, `budget_summary.csv`,
`paired_comparisons.csv`, `grouping_sensitivity.csv`, `leave_one_group_out.csv`,
`individual_group_differences.csv`, `tuning_changes.csv`, `figures/`, `predictor_cache/`,
`pipeline.log`, `pipeline.pid`, `progress.json`, `completion_status.json`, `exit_code.txt`.

## 재개 경로 수정 이력

최초 nested smoke는 완료했으나 재개 시 Parquet가 제거한 index 때문에 cache identity가 달라지는 오류가 발견됐다. validation row key 일치 검증 후 원래 index를 복원하도록 수정했다. 원래 outputs/smoke 및 중지된 outputs/full 로그·checkpoint를 보존하고, 원본 입력·설정·기존 코드 hash를 검증한 checkpoint만 smoke_v2/full_v2로 복사했다. 각 새 실행의 checkpoint_migration.json에 근거를 기록했다. 과학적 계산·후보·모델 설정은 바뀌지 않았다. 최신 full 경로는 outputs/full_v2다.
