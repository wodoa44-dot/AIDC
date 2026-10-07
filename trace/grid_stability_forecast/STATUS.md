# 프롬프트 진행도 (2026-09-27)

## 완료된 코드·산출물

| 범위 | 상태 | 근거 |
|---|---|---|
| 기존 모델 및 데이터 audit | 완료, 일부 과거 설정은 근거 부족으로 unavailable | `results/actual_20260926/current_model_audit.csv`, `data_inventory.csv` |
| 방법론·관측/제어 요구사항 | 문서화 완료 | `methodology.md`, `control_requirements.csv`, `observability_matrix.csv`, `phase0_protocol.md` |
| 완료 bin 기준 target, gap 처리, grouped split | 구현·smoke 검증 | `build_targets.py`, `tests/test_core.py`, `results/actual_20260926/verification.json` |
| +1 s 15-quantile pilot, S1/S2, event-window 확률, 이상화 replay | 완료 | `results/actual_20260926/report_ko.md` |
| 0.2–2.0 s 10-horizon 고정 설정 평가·F1–F3 | 완료 | `results/grid_fixed_20260926/report_ko.md` |
| 5/10/30 s 장기 horizon 고정 설정·별도 그림 | 완료·세 cell 검증 | `results/long_fixed_20260927/report_ko.md` |
| S2 항목별 제거, 5회 run-level metadata permutation, 학습 run 수 곡선 | 파일럿 완료 | `results/ablation_static_20260926/`, `results/actual_20260926/` |
| GPU×workload 조합 밖 평가 | 고정 설정·3 seed 회고적 평가 완료 | `results/unseen_condition_20260926/report_ko.md` |
| grouped outer 5-fold / inner 3-fold, controlled 및 별도 tuning | 5개 fold 완료·hash 검증 | `results/nested_cv5_20260926/progress.json`, `report_ko.md` |
| NLR training recovery 재측정 | offline label 재측정 완료 | `results/event_v2_nlr_training_20260926/report_ko.md` |
| NLR training +1 s CPU+GPU metadata 예측 | 41 job 별도 파일럿; 8 test job | `results/nlr_training_pilot_20260927/report_ko.md` |
| highres 원본 수집일·조합 관계 | 32개 원본 timestamp audit 완료; 4일과 4조합 일대일 | `results/actual_20260926/highres_session_report_ko.md` |
| nested CV 수집일 민감도 | 저장 예측의 날짜별/날짜 제외 기술 통계 | `results/nested_cv5_20260926/day_sensitivity_ko.md` |
| highres event 확률의 일대일 step-onset 매칭 | 탐색 평가 완료; 학습 label과 onset 정의 차이 명시 | `results/actual_20260926/onset_report_ko.md` |
| NLR online-rate 동적 로그 추출 및 B1–B4 | 단일 수집 그룹의 시간순 탐색 완료 | `results/online_runtime_pilot_20260927/report_ko.md` |
| T1–T6, F1–F8 및 원자료 | 파일럿 범위로 생성 | `results/actual_20260926/tables/`, `figures/`, `figure_data/` |

## 이번에 확인·추가한 내용

5-fold runner는 2026-09-27 06:45:17 UTC에 완료했다. `summarize_nested_cv.py`는 검증된 fold의 저장 예측 hash를 확인한 뒤 각 run의 seed 반복을 먼저 평균하고 run에 동일 가중치를 주어 전이·tail·80% coverage와 quantile별 선택 설정을 재생성한다. 32 run에서 history-only CRPS 406.85 W, S2 controlled 379.16 W (gain 6.81%, paired run bootstrap 95% CI 1.75–10.96%), S2 independently tuned 369.39 W (9.21%, 5.79–12.44%)다. 80% coverage는 각각 0.714/0.725/0.715로 부족하다. 이 평가는 기존 데이터를 다시 사용한 회고적 결과이며 새 외부 확인 자료가 아니다.

추가 완료 결과: NLR training +1초 별도 파일럿에서 history-only 705.66 W, model+node 676.66 W로 CRPS가 기술적으로 4.11% 낮았으나 80% coverage는 약 0.89이며 test 조건당 job 하나다. NLR online-rate 서버 로그 6,044개 중 분석창에 3,592개(200 run 모두)가 정렬됐다. 같은 수집 그룹 안의 시간순 150/50 run 파일럿에서 B1/B2/B3/B4 CRPS는 각각 34.25/35.32/34.04/34.29 W였고 B3 coverage는 0.714로 B1 0.760보다 낮았다. 독립 session CI는 계산하지 않았다. Highres step-onset 일대일 매칭은 10개 test run에서 rise/drop miss와 시간당 false alarm을 기록했지만 분류기 학습 label과 사건 정의가 달라 탐색 결과다. 장기 horizon S2 gain은 5/10/30초에서 +1.54/+15.54/-7.02%로 방향이 일정하지 않다.

## 남은 연구와 필요한 입력

- Highres는 네 수집일이 GPU×workload 네 조합과 정확히 결합돼 있다. 날짜/조합 효과 분리 및 독립 session 검증에는 새로운 균형 수집 자료가 필요하다. NLR online-rate 200 run은 하나의 acquisition group이다.
- NLR online-rate에는 약 10초 서버 queue/throughput 로그가 있으나 clock 동기화와 logger window가 미검증이다. 개별 request arrival/latency와 학습 job phase는 여전히 없다. 단일 수집 그룹 파일럿의 B3/B4 수치를 독립 일반화로 해석할 수 없다.
- 0.2–2.0 s 중 +1/+2초와 장기 +5/+10/+30초는 grouped inner 선택을 완료했다. 나머지 0.2–1.8초 horizon은 여전히 고정 설정·단일 seed이며, 전 horizon에서 독립 새 수집 session 검증이 필요하다.
- 사건 정의를 맞춘 별도 onset 모델과 개발 run 내부 threshold 선택은 완료했다. 하강 경보는 100회/시간 시나리오 예산에서 양의 recall을 얻지 못했고, 상승도 test에서 예산을 약간 초과했다. 새 session에서 경보 정책·물리 요구치를 검증해야 한다.
- 센서 availability, 실제 actuator 응답·간섭, PCC 전력·계통 모델, 처리량/SLA. 현재 replay는 명시적 가정의 GPU 전력 대리지표 실험이다.
- 독립 새 데이터에서 calibration 및 metadata 효과 재확인. 현재 CI는 저장 예측의 run 재표집만 포함한다.

## 검증·재현

```bash
PYTHONPATH=trace/grid_stability_forecast python -m unittest discover -s trace/grid_stability_forecast/tests -v
python trace/grid_stability_forecast/summarize_nested_cv.py
python trace/grid_stability_forecast/evaluate_onset.py
python trace/grid_stability_forecast/extract_online_runtime.py
python trace/grid_stability_forecast/online_runtime_pilot.py
python trace/grid_stability_forecast/summarize_online_runtime.py
python trace/grid_stability_forecast/audit_highres_sessions.py
python trace/grid_stability_forecast/assess_day_sensitivity.py
python trace/grid_stability_forecast/run_nlr_training_pilot.py
python trace/grid_stability_forecast/summarize_nlr_training.py
python trace/grid_stability_forecast/run_long_horizon.py
python trace/grid_stability_forecast/summarize_long_horizon.py
cat trace/grid_stability_forecast/results/nested_cv5_20260926/progress.json
```

재학습은 `python trace/grid_stability_forecast/run_nested_cv.py`로 재개한다. 기존 hash가 일치하면 검증된 fold를 재사용한다. 실제 GPU cap이나 시설 명령은 수행하지 않는다.

## 2026-09-27 추가 완료

- NLR training job-grouped 4-fold×3-seed 고정 설정 평가를 완료·검증했다. 41 job의 균형 평균 CRPS는 history-only 613.71 W, model+node 607.11 W로 기술적 개선 1.08%(41개 중 28개 job 개선)다. 80% coverage는 각각 약 0.89/0.90이며 원본 session 독립성은 미확인이다. 근거: `results/nlr_training_cv4_20260927/report_ko.md`, `verification.json`.
- Highres +2초의 24개 후보 q50 screening→조건별 6개 15-quantile inner 선택→3-seed outer 평가를 완료했다. 검증 15항목 모두 통과. history 599.87 W, S2 controlled 571.99 W, S2 tuned 522.93 W; tuned 80% coverage 0.669로 낮다. 기존 outer holdout 재사용이며 새 독립 확인이 아니다. 근거: `results/nested_h2000_20260927/report_ko.md`, `verification.json`, `progress.json`.
- Future 1초 step-onset label로 별도 이진 분류기를 학습·일대일 매칭했다. history-only 기준 drop recall 0.565, rise recall 0.425; 시간당 false alarm은 491/371로 높다. 운영용 alarm threshold가 아니다. 근거: `results/onset_aligned_20260927/report_ko.md`.

남은 일은 각 horizon의 독립 자료 재검증, 실측 장치·PCC 제어 반응, session/clock 독립성 확인, calibration과 거짓 경보 억제다. 현재 저장소에 게시할 작은 결과 요약은 원본 데이터·예측 parquet·학습 모델을 포함하지 않는다.

## +2초 calibration 사후 진단

`diagnose_calibration.py`는 +2초 nested 실험의 저장 예측 hash를 확인하고 quantile별 empirical coverage와 미래 ΔP 기준 steady/rise/drop 층의 80% 구간 coverage를 run 균형으로 산출한다. S2 tuned의 coverage는 steady/rise/drop 각각 0.657/0.672/0.683, 평균 절대 quantile gap은 0.0365로 history-only 0.0211보다 크다. 이 결과를 사용해 test에서 보정값을 학습하거나 설정을 고르지 않았다. 근거: `results/nested_h2000_20260927/calibration_diagnostic_ko.md`, `calibration_reliability.png`, `calibration_transition.png`.

## Onset 문턱값 내부 선택

`tune_onset_threshold.py`는 기존 개발 22 run에서 4 run(각 GPU×workload 조합 1개)을 검증용으로 빼고, history-only 분류기와 고정 threshold grid를 평가했다. 시간당 오경보 100회 이하 시나리오를 적용하면 rise threshold 0.70(검증 recall 0.297, 오경보 80/h)이 선택됐고 기존 test 확률에서 recall 0.302, 오경보 103/h였다. Drop은 검증 예산을 만족하는 사용 가능한 threshold가 없어 0.99에서 경보 0회·recall 0으로 끝났다. 따라서 현재 모델로 하강 경보 운영 가능성을 보이지 못했다. 기준 0.5 재평가의 event/경보/매칭/오경보 정수는 기존 평가와 정확히 일치한다. 근거: `results/onset_threshold_20260927/report_ko.md`, `validation_threshold_tradeoff.png`. 예산과 100 W event는 시나리오 값이며 실제 계통 요구치가 아니다.

## Highres +5초 grouped nested 선택 완료

`run_nested_horizon.py`를 기존 `long_fixed_20260927/h5000` 입력에 적용해 동일 24개 후보·조건별 상위 6개 full15·3 seed outer 평가를 완료했다. `results/nested_h5000_20260927/progress.json`의 15개 검증 항목이 모두 참이다. 10개 재사용 test run의 균형 CRPS는 history 931.15 W, S2 controlled 931.09 W(기술적 gain +0.007%, paired run bootstrap 95% CI −5.54~+6.65%), S2 tuned 858.10 W(+7.85%, +1.85~+15.39%)다. 80% 구간 coverage는 각각 0.649/0.650/0.731이다. 이 CI는 기존 test run 재표집이며 새로운 독립 수집 session 불확실성을 포함하지 않는다. `calibration_diagnostic_ko.md`는 미래 ΔP별 coverage와 quantile reliability를 별도 기록한다. 따라서 5초 controlled metadata의 유용성은 판정 불충분이고, separately tuned track의 개선은 회고적·보정 불충분 상태에서 관찰된 것이다.

## Highres +10초 grouped nested 선택 완료

`results/nested_h10000_20260927/`에서 15개 검증 항목을 모두 통과했다. 10개 재사용 test run·3 seed의 균형 CRPS는 history 838.79 W, S2 controlled 756.42 W(gain +9.82%, paired run bootstrap 95% CI −1.92~+19.70%), S2 tuned 734.79 W(+12.40%, −0.71~+22.77%)다. 두 CI 모두 0%를 포함하므로 효과의 방향은 이 재사용 holdout에서 불확실하다. 80% coverage는 각각 0.671/0.659/0.704로 명목 0.8보다 낮다. 근거: `report_ko.md`, `calibration_diagnostic_ko.md`, `verification.json`.

## Highres +30초 및 horizon 비교 완료

`results/nested_h30000_20260927/`에서 15개 검증 항목을 모두 통과했다. 균형 CRPS는 history 839.93 W, S2 controlled 901.74 W(gain −7.36%, paired run bootstrap 95% CI −18.55~+1.89%), S2 tuned 813.55 W(+3.14%, −6.66~+11.99%)다. 80% coverage는 0.690/0.693/0.760이다. 두 paired CI 모두 0%를 포함해 효과 방향이 불확실하다.

`compare_nested_horizons.py`로 +1/2/5/10/30초의 검증된 동일 highres holdout 결과를 표·꺾은선 그래프로 묶었다. 근거: `results/nested_horizon_comparison_20260927/report_ko.md`, `nested_horizon_scores.png`, `nested_horizon_gain.png`. S2 tuned의 CRPS는 다섯 horizon에서 모두 history보다 낮았지만, controlled S2는 +30초에서 악화됐고 대부분의 horizon에서 80% coverage가 명목치보다 낮다. 모두 같은 기존 holdout을 재사용했으며 원본 수집 session 독립성은 미확인이다.
