# AIDC 예측모델 pilot

GPU 합산 전력의 1초 endpoint q50만 평가하는 소규모 실험이다. 제어·MPC·저장장치·계통 시뮬레이션은 실행하지 않는다. 프로젝트 root는 실행 위치와 classification manifest를 대조해 찾으며 config.yaml에는 프로젝트 상대 경로를 저장한다.

## 완료 결과

`outputs/pilot_v1/`에 24회 q50 학습 결과가 있다. 초소형 smoke는 `outputs/smoke_v1/`의 100행·5 trees·1 fit이며 성능표에서 제외했다. 최종 학습은 개발 23개 run의 6,842행, 보류 평가는 9개 run의 4,324행이다. run당 training 300행/evaluation 600행 cap을 확인했다. 최초 전체 실행은 22.098초, 24회 fit 및 prediction 합계는 2.438초다. report-only 재생성은 6.871초이며 원래 실행 로그와 상태는 `.previous` 파일로 보존된다.

[한국어 보고서](outputs/pilot_v1/report_ko.md)와 [HTML](outputs/pilot_v1/report_ko.html)에 표·그림·해석이 있다. PNG 300dpi/SVG 7쌍과 plotting CSV를 저장했다. 실제 HTML 렌더링에서 이미지 로딩과 가로 넘침을 확인했다. `report-only` 재생성에서 새 fit이 발생하지 않았다.

개발 OOF 선택은 residual target + 5초 history다. 보류 MAE는 Persistence보다 11.22% 낮았으나 개발 날짜 평균에서는 Persistence보다 높았다. Metadata는 선택 모델 대비 보류 MAE를 악화시키고 p95 오차를 낮췄다. 날짜·구간·지표별 차이를 먼저 확인하고, 하나의 보류 날짜를 일반화하지 않는다. 추가 모델이나 대규모 실험은 자동 시작하지 않는다.

## 실행과 재개

아래 명령은 이 pilot 폴더에서 실행한다. manage.py가 확인된 Python·작업 디렉터리·config·로그의 절대 경로로 nohup을 시작한다. stdin은 /dev/null, Python은 unbuffered, 모델은 순차 실행이며 LightGBM threads는 4다.

```bash
/usr/bin/python -m unittest -v test_pilot.py
/usr/bin/python manage.py launch --config config.yaml --output-dir outputs/new_smoke --smoke
/usr/bin/python manage.py launch --config config.yaml --output-dir outputs/new_pilot
/usr/bin/python manage.py status --config config.yaml --output-dir outputs/pilot_v1
tail -f outputs/pilot_v1/pipeline.log
/usr/bin/python manage.py stop --config config.yaml --output-dir outputs/pilot_v1
/usr/bin/python manage.py launch --config config.yaml --output-dir outputs/pilot_v1 --resume
/usr/bin/python manage.py launch --config config.yaml --output-dir outputs/pilot_v1 --report-only
```

완료된 프로세스는 stop할 대상이 없으며 도구가 signal 전송을 거부한다. 재개 시 코드·입력·config identity와 표본 및 fit checkpoint checksum이 일치해야 한다. 변경된 구성은 새 output을 사용한다. `.previous`는 이전 실행 로그·상태 기록이다. `--report-only`는 모델 재학습 없이 저장 prediction 집계 및 저장 모델의 짧은 연속 구간 inference와 렌더링만 수행한다.

## 파일 안내

- `data.py`: 연속 bin feature, 1초 간격 후보, 시간 층화 추출, fold별 encoding 및 run 가중치.
- `pipeline.py`: 기존 quantile 모델 정렬 감사, 18개 target/history 개발 fit, 3개 metadata fit, 최대 3개 final fit, checkpoint.
- `report.py`, `render_html.py`: 예측 지표·표·PNG/SVG·한국어 보고서와 실제 HTML 렌더링.
- `manage.py`: nohup 실행, 중복 방지, PID identity 검증, status/stop/resume.
- `test_pilot.py`: feature 인과성, 단절·결측, 고정 추출, 가중치, unseen encoding, Persistence 및 p99 검증.
- `outputs/pilot_v1/run_inventory.csv`, `excluded_runs.csv`: 32개 후보 run과 제외 사유. 이번에는 모두 최소 기준을 통과했다.
- `train_anchors.parquet`, `evaluation_anchors.parquet`: 공통 표본 key·추출 규칙. timestamp는 표준화 데이터와 같은 run 시작 기준 상대 초 단위다. target은 anchor+1초에 끝나는 200ms bin이다.
- `feature_schema.json`: 현재를 포함한 5/25/50개 bin. history 창 바로 밖의 lag_5/25/50은 각각 1/5/10초 구성에서 제외한다. 기존 q50 감사는 기존 경계 lag를 포함한 원래 정의를 복원한다.
- `legacy_audit_development/`, `legacy_audit_holdout/`: 원래 q50 및 15 quantile 정렬 median 감사. 파일 hash·학습 encoding·sample key·target·저장 median을 대조했다. 기존 30초 eligibility와 새 10초 공통 anchor 차이로 겹치는 표본만 사용하며 새 pilot 결과와 별도다.
- `sorting_effect_per_run.csv`, `sorting_effect_summary.csv`: 위 감사 결과의 정렬 개선·악화 및 Persistence 대비 악화 run 수 추가 집계.
- `aggregate_metrics.csv`, `per_run_metrics.csv`, `per_day_metrics.csv`, `endpoint_region_counts.csv`: run별 계산 후 동일 가중 집계. 큰 endpoint 변화는 q90 초과이며 미래 경로 최대 ramp가 아니다. p99는 100표본 미만 run에서 N/A다.
- `clipping_audit.csv`: raw가 primary이며 0 하한 적용은 모든 모델에 같은 진단으로만 산출한다.
- `encoding_diagnostics.csv`: training-only 범주 mapping, unseen/결측률/상수 열. 숫자 결측은 추정하지 않는다.
- `sampling_region_audit.csv`: training에서 결정한 threshold로 추출 전후 구성비를 비교하며 이를 보고 재추출하지 않는다.
- `fit_summary.json`, `identity.json`, `sampling_checkpoint.json`, `resolved_config.yaml`: 실제 fit 수·시간·학습량·출처·전체 API parameter 및 재사용 identity.

날짜 group의 독립 수집 session은 확인되지 않았다. 날짜와 hardware/workload가 교락되어 있고 10초 history가 겹친다. Bootstrap·정식 유의성 검정·새 모델 CRPS/coverage는 산출하지 않았다. 작은 training cap의 실패로 모델 계열 전체의 한계나 예측 불가능성을 주장하지 않는다. 기존 실험과 `control_models`는 수정하지 않는다.
