# Residual·metadata 후속 개발 실험

고정 5초 history와 LightGBM q50으로 A/R/RM × cap100/300/600 × 개발 3 folds를 비교했다. 본 신규 27 fits, 재사용 0 fits, 별도 smoke 1 fit가 완료됐다. 09-22는 기존 저장 예측 재분석만 수행했다. 제어·MPC·계통 실행, history/seed/모델 탐색, 최종 전체 개발 refit은 없다.

[한국어 보고서](outputs/full_v1/report_ko.md), [HTML](outputs/full_v1/report_ko.html), [검증](outputs/full_v1/verification.json).

Run-balanced R MAE는 769.754→693.568→690.984 W, Persistence는 521.061 W다. R은 A보다 전체 평균이 모든 cap에서 낮지만 날짜별 예외가 있다. R이 Persistence를 넘은 날짜/cap은 09-20/300뿐이다. Metadata 평균 MAE 효과는 +2.732/−4.130/−12.928 W, p95 효과는 모두 개선·p99는 모두 악화다. 독립 최종 검증이나 인과 결과로 해석하지 않는다.

## 실행

프로젝트 root `/home/work/AIDC`에서 실행한다. 기존 완료 출력은 보존하며, 새 실행은 새로운 output 경로를 사용한다. 동일 설정 중복 실행을 관리 도구와 pipeline flock으로 차단한다. stdin은 /dev/null, Python unbuffered, nohup 백그라운드 실행이다. 출력마다 PID·시작 시각·명령행·로그·progress·exit code·단계 및 fit checkpoint가 있다.

```bash
/usr/bin/python experiments/prediction_models/aidc_residual_metadata_followup_20261001/manage.py launch --config experiments/prediction_models/aidc_residual_metadata_followup_20261001/config.yaml --output-dir experiments/prediction_models/aidc_residual_metadata_followup_20261001/outputs/new_smoke --smoke
/usr/bin/python experiments/prediction_models/aidc_residual_metadata_followup_20261001/manage.py launch --config experiments/prediction_models/aidc_residual_metadata_followup_20261001/config.yaml --output-dir experiments/prediction_models/aidc_residual_metadata_followup_20261001/outputs/new_full
```

중단된 실험은 같은 config/output과 `--resume`으로 재개한다. 코드·입력·config identity, anchor/encoded feature/weight hash 및 fit 산출물 checksum이 같을 때만 checkpoint를 사용한다. 실행 중 stop은 PID start ticks와 argv identity를 검증한다. 완료된 PID에는 signal을 전송하지 않는다.

```bash
/usr/bin/python experiments/prediction_models/aidc_residual_metadata_followup_20261001/manage.py status --output-dir experiments/prediction_models/aidc_residual_metadata_followup_20261001/outputs/full_v1
/usr/bin/python experiments/prediction_models/aidc_residual_metadata_followup_20261001/manage.py stop --output-dir experiments/prediction_models/aidc_residual_metadata_followup_20261001/outputs/full_v1
```

## 학습 없이 최종 보고서 재생성

아래 첫 명령이 끝나 completion_status.json에서 exit_code=0을 확인한 뒤, 두 번째와 세 번째를 실행한다. 별도 finalize는 저장 모델 inference와 독립 원본 대조·질문별 수치 보충만 수행하며 fit을 하지 않는다. 이전 실행 로그/상태는 `.previous`로 보존한다. 원래 본 실행 시간은 timing.json, report-only 시간은 report_only_timing.json에 분리된다.

```bash
/usr/bin/python experiments/prediction_models/aidc_residual_metadata_followup_20261001/manage.py launch --config experiments/prediction_models/aidc_residual_metadata_followup_20261001/config.yaml --output-dir experiments/prediction_models/aidc_residual_metadata_followup_20261001/outputs/full_v1 --report-only
/usr/bin/python experiments/prediction_models/aidc_residual_metadata_followup_20261001/tools/verify_finalize.py experiments/prediction_models/aidc_residual_metadata_followup_20261001/outputs/full_v1
/usr/bin/python experiments/prediction_models/aidc_forecast_pilot_20261001/render_html.py experiments/prediction_models/aidc_residual_metadata_followup_20261001/outputs/full_v1
```

`pipeline.py`는 pilot의 원본 feature cache와 `data.py`의 feature/encoding/weight 구현을 읽어 사용한다. 원본 데이터는 복제하지 않는다. 본 cap300 A/R/RM 예측이 원래 pilot 저장 예측과 일치함을 별도 재현 검증했다. 파생 표본은 새 폴더에 저장한다. 10초 anchor eligibility를 유지하되 feature는 5초 이내만 사용한다. 각 fit은 날짜/run 완전 분리와 run별 총 가중치 동일성을 검증한다.

`fit_registry.csv`는 fit 생성 이력을 기록한다. Resume에서 checkpoint를 읽는 것은 pilot 모델 재사용과 구분한다. 표본 증가가 독립 run/조건 증가를 뜻하지 않는다. 23개 개발 run에 13,338개 공통 평가 anchor이며 cap별 고유 training 시점은 2,300/6,842/13,338개다. 각 run은 두 training fold에서 쓰인다.
