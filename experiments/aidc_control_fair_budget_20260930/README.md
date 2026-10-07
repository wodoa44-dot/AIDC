# 동일 장치별 예산 제어 실험

E1 가정 기반 독립 구현. 기존 평가셋 재사용 분석이다. 원본 파일은 읽기만 한다.
`PROMPT.md`에 사용자 요구사항을 보존했다. `config.yaml`은 test 결과 확인 전 고정했다.

## 범위

기존 seed 17, horizon 1초, P2/P4 q0.5 outer OOF 예측을 고정한다.
내부 grouped-validation 예측과 모델 파일이 없어 controller tuning 비교는 blocked다.
다른 outer fold 예측으로 tuning하면 test 정보가 유입될 수 있어 사용하지 않는다.
실행 가능한 전체 비교는 공통 파라미터를 고정한 C0~C5 × 예산 5개이다.
완료 상태는 `completed_with_blocked_scope`, `all_completed=false`로 구분한다.

60초 완전 창의 40초 제어+20초 공통 SoC 복구를 평가한다. 불완전 창 제외는 기록한다.
장치별 AC 절대 처리량 예산은 outer-training C2 run-balanced 평균 출력으로 정한다.
두 장치의 종료 SoC와 출력까지 검증한다. 복구 ramp·처리량도 포함한다.
통계는 run 내 창별 ramp를 연결 없이 모아 분위수를 계산한 후 run-balanced 집계한다.
2,000회 source-group bootstrap은 source-group 독립성 미검증 및 고정 정책 조건부이다.

## 실행

```bash
cd /home/work/AIDC/experiments/aidc_control_fair_budget_20260930
/usr/bin/python -m unittest -v test_control.py
/usr/bin/python manage.py launch --config "$PWD/config.yaml" --output-dir "$PWD/outputs/smoke" --smoke
/usr/bin/python manage.py launch --config "$PWD/config.yaml" --output-dir "$PWD/outputs/full"
```

`manage.py launch`는 실제 `nohup bash /absolute/path/run_pipeline.sh ...`를 실행한다.
stdin=/dev/null, stdout/stderr=pipeline.log, start_new_session, Python -u를 사용한다.
같은 config hash의 `/proc` 명령행을 검사하고 실행 잠금으로 중복 실행을 막는다.
runner는 identity 확인 → audit → calibration → control → aggregation → figures/report를 순차 실행한다.
오류가 나면 다음 단계는 실행하지 않고 nonzero exit code와 실패 상태를 기록한다.

```bash
tail -f outputs/full/pipeline.log
cat outputs/full/progress.json
/usr/bin/python manage.py status --config "$PWD/config.yaml" --output-dir "$PWD/outputs/full"
/usr/bin/python manage.py stop --config "$PWD/config.yaml" --output-dir "$PWD/outputs/full"
/usr/bin/python manage.py launch --config "$PWD/config.yaml" --output-dir "$PWD/outputs/full" --resume
```

stop은 PID/start-time/명령행 확인 후 pidfd로 해당 프로세스에만 TERM을 보낸다.
완료 task는 run/controller/budget 단위 checkpoint이며 중단 task만 다시 계산한다.
resume 시 config, 코드, 입력, 모델 예측 및 모델 선택 SHA256 identity를 비교한다.
다르면 별도 output을 사용해야 한다. 이전 로그와 상태 파일은 `.previous`로 보존한다.
SIGKILL/재부팅은 상태 기록이 불가능하다. status의 실제 프로세스 확인 후 checkpoint로 재개한다.
`nohup` 자체는 재부팅 복구를 제공하지 않는다.

산출물: `report_ko.md`, `implementation_audit.md`, `resolved_config.yaml`, `provenance.json`,
`fair_budget_metrics.csv`, `per_run_metrics.csv`, `budget_summary.csv`, `paired_comparisons.csv`,
`constraint_checks.csv`, `device_assumptions.csv`, `figures/`, `checkpoints/`,
`pipeline.log`, `pipeline.pid`, `progress.json`, `completion_status.json`, `exit_code.txt`.
