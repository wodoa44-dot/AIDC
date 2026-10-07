# `model_200ms_1s_2s` 파일 안내서

이 문서는 지금까지의 대화에서 생성·이동·검증한 `trace/model_200ms_1s_2s` 패키지와 산출물의 용도를 설명한다. 해시 이름으로 생성된 수백 개의 checkpoint와 반복 모델 파일은 개별 해시 대신 파일명 패턴과 디렉터리 단위로 설명한다. 사람이 직접 판단에 사용하는 보고서, 표, 예측, 모델 번들은 파일별로 설명한다.

## 가장 먼저 볼 파일

- [RESULTS_INDEX.md](RESULTS_INDEX.md): 최종 결과와 오차 그래프 바로가기.

1. `README.md`: 최초 lightweight 모델의 실행법과 패키지 개요.
2. `outputs/lite_full_20260920/report_en.md`: 최초 lightweight 모델의 학습·평가 결과.
3. `validation/audit_20260921_full_v3/report_en.md`: 최초 모델을 고정한 상세 후향 audit의 최종본.
4. `improvements/failure_aware_20260921_v1/artifacts/full/report_en.md`: failure-aware 재학습 실험의 최종 영문 보고서.
5. `improvements/failure_aware_20260921_v1/artifacts/full/final_selections.csv`: 새 모델의 1초·2초 최종 선택.
6. `improvements/failure_aware_20260921_v1/validation/anomaly_audit_20260921_v1/summary_ko.md`: 신규 모델 이상치 분석의 한국어 요약.
7. `improvements/failure_aware_20260921_v1/validation/anomaly_audit_20260921_v1/worst_cases.csv`: 신규 모델의 최대 저예측·과대예측 등 대표 사례.
8. `improvements/failure_aware_20260921_v1/models/power-forecast-failure-aware-20260921-v1/history_meta_full15_deployment.joblib`: 신규 metadata 배포 후보 번들.

## 루트 코드와 설정

- `README.md`: 최초 lightweight 모델의 데이터 범위, 학습·평가·export·replay 명령을 설명한다.
- `configs/lite_20260920.json`: 최초 lightweight 실험의 history, horizon, LightGBM, seed, sampling 설정.
- `train.py`: 최초 후보 탐색, grouped validation, 최종 refit을 수행하는 학습 진입점.
- `finalize.py`: 최초 실험 결과를 집계하고 보고서·export 산출물을 완성한다.
- `predict.py`: 저장된 deployment bundle로 추론하는 CLI/API 진입점.
- `replay.py`: 저장된 모델을 연속 시계열에 다시 적용하여 offline 결과와 비교한다.
- `benchmark.py`: 모델 로드 시간, 추론 지연시간, 크기와 프로세스 메모리를 측정한다.
- `src/core.py`: 데이터 구성, causal feature, quantile 학습·정렬, scoring 등 공통 핵심 로직.
- `src/online.py`: 실시간/연속 replay용 상태 관리와 online feature 처리 로직.
- `src/__init__.py`: `src` Python 패키지 표시 파일.
- `tests/test_core.py`: causal feature, quantile 처리, split 및 공통 로직의 단위 테스트.
- `__pycache__/`, `src/__pycache__/`, `tests/__pycache__/`: Python이 자동 생성한 bytecode cache로, 분석 결과나 배포 파일이 아니다.

## 최초 lightweight 학습 결과: `outputs/lite_full_20260920/`

이 디렉터리는 최초 absolute-power lightweight Quantile LightGBM 실험의 full run 결과다. 현재 regular file 약 844개이며, 대부분은 resumable checkpoint와 fold별 evaluation cache다.

### 설정·상태·재현성

- `effective_config.json`: 실제 실행에 사용된 병합 완료 설정.
- `configuration_signature.json`: resume 시 설정 호환성을 검사하는 configuration hash.
- `environment.json`: Python 및 주요 라이브러리 버전.
- `source_manifest.csv`: 입력 source와 provenance 목록.
- `source_hashes.csv`: 실행 코드·설정 파일의 hash.
- `run_inventory.csv`: 사용 가능한 canonical run과 cohort 정보.
- `development_runs.csv`: 개발 데이터에 포함된 run 목록과 메타데이터.
- `reused_holdout_runs.csv`: 최종 개발 학습에서 제외한 4개 reused-holdout run.
- `development_folds.csv`: 최종 선택용 grouped development fold 정의.
- `corrected_folds.csv`: grouped-outer 및 inner split의 수정·확정된 fold 정의.
- `metadata_split_usage.csv`: metadata encoder가 어느 train split에서 fit됐는지 기록한다.
- `power_feature_schemas.csv`: horizon/history 조건별 입력 feature 순서와 의미.
- `execution.log`: 학습·평가 실행 기록.
- `status.json`: 실행 단계, 시작·종료 시간과 완료 여부.

### 후보 탐색·선택

- `outer_candidate_scores.csv`: 각 outer fold의 inner validation 후보 점수.
- `final_candidate_scores.csv`: 전체 development grouped validation 후보 점수.
- `outer_selections.json`: outer fold마다 선택된 history/tree 설정.
- `final_selections.json`: 최종 development 선택 결과.
- `outer_evaluation_tasks.csv`: fold·seed·condition별 실제 evaluation task와 artifact 경로.
- `tree_diagnostics.csv`: 개별 quantile 모델의 tree 수, 학습 상태와 진단값.
- `checkpoints/candidate_<hash>.json`: 후보 하나의 완료 상태와 validation 점수. 설정 hash 기반 resume 단위다.
- `checkpoints/evaluation_<hash>.json`: evaluation task 하나의 완료 상태와 예측 경로.
- `evaluation/<hash>/`: fold·seed·condition 단위의 임시/재사용 가능한 모델 및 예측 cache.
- `training_samples/*.npz`: outer/inner split별 선택된 train row identity와 sampling 결과. `development_*`는 최종 선택용, `outerN_innerN_*`는 nested validation용이다.

### 예측·평가 표

- `outer_predictions.parquet`: grouped-outer test의 target별 full-quantile 예측.
- `reused_holdout_predictions.parquet`: 최종 development 모델의 4개 reused-holdout 예측.
- `replay_predictions.parquet`: deployment bundle을 reload해 연속 replay한 예측.
- `aggregate_metrics.csv`: horizon·condition별 종합 CRPS/MAE 등.
- `comparison_summary.csv`: history-only와 history+metadata의 핵심 비교 요약.
- `per_run_metrics.csv`: run별 성능.
- `per_run_meta_gain.csv`: 같은 run에서 metadata 추가 전후의 paired gain.
- `per_cohort_metrics.csv`: GPU/workload cohort별 성능.
- `per_seed_fold_metrics.csv`: seed와 outer fold별 성능.
- `transition_metrics.csv`: rising/falling/stable endpoint별 요약.
- `confidence_intervals.csv`: run-level 비교의 불확실성 구간.

### 배포·검증·보고

- `deployment_manifest.json`: export된 deployment bundle과 포함 모델 정보.
- `deployment_export_metrics.csv`: compact deployment export의 holdout MAE·coverage·full15 차이.
- `verification.json`: reload/replay 일치 여부와 주요 무결성 검사.
- `benchmark.json`: 최초 배포 bundle의 크기·load·추론 latency·memory 측정.
- `report_en.md`: 최초 lightweight 실험의 최종 영문 보고서.
- `plots/accuracy_size_latency.png`: 정확도·모델 크기·latency 비교.
- `plots/lite_vs_reference_crps.png`: lightweight와 reference CRPS 비교.
- `plots/observed_and_forecast.png`: 관측값과 예측 시계열 예시.
- `plots/per_run_meta_gain.png`: run별 metadata gain.

## 최초 모델 번들: `models/power-forecast-lite-20260920/`

- `full_h1000_history_only.joblib`: 1초 history-only, 15 quantile 전체 모델.
- `full_h1000_history_meta.joblib`: 1초 history+metadata, 15 quantile 전체 모델.
- `full_h2000_history_only.joblib`: 2초 history-only, 15 quantile 전체 모델.
- `full_h2000_history_meta.joblib`: 2초 history+metadata, 15 quantile 전체 모델.
- `history_only_deployment.joblib`: 두 horizon의 compact deployment 모델을 묶은 history-only bundle.
- `history_meta_deployment.joblib`: 두 horizon의 compact deployment 모델을 묶은 metadata bundle.
- `h1000_history_only/feature_schema.json`, `h1000_history_meta/feature_schema.json`: 1초 조건별 feature schema.
- `h2000_history_only/feature_schema.json`, `h2000_history_meta/feature_schema.json`: 2초 조건별 feature schema.
- 각 조건의 `training_identities_weights.npz`: 최종 학습 row identity와 weight. 학습 재현성과 leakage 확인용이다.

## 최초 모델 상세 audit: `validation/`

- `validate.py`: 동결된 최초 모델과 저장 예측을 대상으로 continuous retrospective audit를 만드는 실행 코드.
- `metrics.py`: error distribution, interval score, CRPS 근사, transition, episode segmentation 등의 순수 metric 함수.
- `tests/test_metrics.py`: pairing, error sign, quantile metric, episode boundary 등의 테스트.

### 권위본: `validation/audit_20260921_full_v3/`

`full_v3`가 최초 모델 audit의 최종 권위본이다. `audit_20260921_full`, `full_v2`는 그 이전 반복 실행이며, `audit_20260921_smoke*`는 소규모 실행 검증용이다.

- `validation_spec.json`: audit 전에 고정한 평가 정의, threshold, bootstrap 설정.
- `evaluation_inventory.csv`: assessment/run/segment/cohort/model/fold/seed/representation별 평가 범위와 coverage.
- `data_integrity_checks.json`: horizon identity, 중복, paired target, leakage/provenance 검사 결과.
- `aggregate_reconciliation.csv`: 기존 발표 집계와 audit 재계산값의 일치 검사.
- `per_target_evaluation.parquet`: audit의 canonical target별 예측·관측·raw/corrected quantile 표.
- `per_run_metrics.csv`: run별 CRPS, MAE, RMSE, bias, tail, coverage, persistence 비교.
- `per_cohort_metrics.csv`: GPU/workload cohort별 target-weighted 및 run-balanced 지표.
- `tail_error_metrics.csv`: p50/p90/p95/p99/max와 worst-5% 오류.
- `tolerance_exceedance.csv`: 50/100/200/300/500/750/1000 W threshold 초과율과 episode 요약.
- `failure_episodes.parquet`: 각 threshold에서 연속 200 ms 발행으로 구성한 개별 오류 episode.
- `transition_metrics.csv`: stable/rising/falling별 오류·bias·tail·CRPS.
- `change_confusion_matrices.csv`: 실제 endpoint-change class와 예측 change class의 confusion matrix.
- `interval_diagnostics.csv`: 80%/90% interval coverage, width, violation, interval score와 quantile calibration.
- `export_effects.csv`: raw q50, full-15 정렬, compact-3 정렬의 paired 영향.
- `paired_uncertainty.csv`: 2,000회 paired run bootstrap 구간.
- `worst_cases.csv`: 고정 규칙으로 선택한 최대 오류·최장 episode·export 변화 사례.
- `source_artifact_hashes.csv`: audit가 읽은 모델·예측·코드의 hash.
- `execution_commands.txt`: 정확한 smoke/full 재실행 명령.
- `execution.log`: audit 실행 기록.
- `completion_status.json`: 완료·통과 여부와 artifact 경로.
- `report_en.md`: 평균 성능, run/cohort 일관성, rare/sustained error, transition, calibration, export 효과를 설명한 최종 보고서.
- `figures/per_run_metadata_gain.png`: cohort별 run metadata gain.
- `figures/absolute_error_ecdf.png`: 모델·baseline의 절대오차 누적분포.
- `figures/tolerance_exceedance.png`: threshold별 초과율.
- `figures/episode_duration_ecdf.png`: 오류 episode 지속시간 분포.
- `figures/coverage_by_transition.png`: horizon·transition별 interval coverage와 violation.
- `figures/full_vs_compact_q50_change.png`: full15와 compact3 postprocessing 차이.
- `figures/timeseries_00.png`–`timeseries_04.png`: 결정론적으로 선택한 worst/typical 시계열 창.

### 이전/검증 실행 디렉터리

- `audit_20260921_full/`: 최초 full audit 결과. 이후 provenance·표현이 보완되어 v3로 대체됐다.
- `audit_20260921_full_v2/`: 두 번째 full audit 결과. v3 이전 중간본이다.
- `audit_20260921_smoke/`, `audit_20260921_smoke_v2/`: 실제 데이터의 일부로 pipeline 동작을 확인한 smoke 결과. 결론 보고용이 아니다.
- 위 디렉터리의 동명 파일은 `full_v3`와 같은 역할이지만 해당 실행 범위/버전의 결과다.

## Failure-aware 재학습: `improvements/failure_aware_20260921_v1/`

이 버전은 absolute, residual, transition-weighted residual 세 family를 실제 학습하고, mean CRPS 선택과 reliability-aware 선택을 비교한 독립 실험이다. 기존 deployment path를 덮어쓰지 않는다.

### 구현 파일

- `README.md`: 테스트, smoke, full, resume, replay, benchmark 명령.
- `config.json`: 12 candidate/horizon, quantiles, seeds, sampling, reliability selection, bootstrap 설정.
- `experiment.py`: diagnosis 이후 grouped-inner/outer search, refit, export까지 수행하는 주 실행기.
- `runtime.py`: causal sample 구성, encoder, LightGBM fit/predict, weighting과 공통 runtime 지원.
- `reporting.py`: old/new prediction 결합, metric·episode·bootstrap·plot·보고서 생성.
- `replay.py`: 신규 full15 deployment bundle의 연속 replay와 batch 결과 비교.
- `benchmark.py`: 신규 30-model bundle의 실제 size/load/prediction/end-to-end latency 측정.
- `failure_diagnosis.md`: 이전 audit evidence를 바탕으로 학습 전에 고정한 실패 유형 진단.
- `tests/test_experiment.py`: residual 복원, train-only weighting, run weight 균등화, split isolation, selection 결정성, episode, quantile 처리, resume 테스트.
- `__init__.py`: Python package 표시 파일.
- `__pycache__/`, `tests/__pycache__/`: 자동 생성 cache이며 결과물이 아니다.

### Full 학습 결과: `artifacts/full/`

현재 regular file 약 952개다. 핵심 표 외 대부분은 checkpoint, evaluation bundle, sampled identity다.

#### 설정·데이터·상태

- `frozen_experiment_spec.json`: 실제 full 실험의 고정 설계와 선택 규칙.
- `configuration_signature.json`: configuration-aware resume hash.
- `environment.json`: 실행 환경과 라이브러리 버전.
- `input_manifest.csv`: 원 입력 source의 경로·크기·hash/provenance.
- `development_runs.csv`: 28개 development run.
- `reused_holdout_runs.csv`: fitting/selection에서 제외한 4개 holdout run.
- `development_inner_split_manifest.csv`: 최종 development selection의 grouped-inner split.
- `outer_inner_split_manifest.csv`: 각 outer-training partition 내부의 grouped-inner split.
- `data_integrity_checks.json`: run 분리, target identity, horizon와 causal construction 검사.
- `feature_schemas.csv`: horizon/history/condition별 정확한 입력 feature 순서.
- `failure_diagnosis.md`: 실험 시작 전에 저장한 진단의 artifact 복사본.
- `execution.log`: full 학습·평가 실행 로그.
- `status.json`: 시작·종료 시각과 `completed` 상태.
- `completion_status.json`: report, model root, replay pass와 artifact 수를 기록한 최종 완료 표식.
- `source_hashes.csv`: 실험 코드·설정·진단 파일 hash.

#### 후보 점수와 선택

- `outer_candidate_fold_metrics.csv`: outer fold 안 각 inner fold의 12개 후보 raw 점수.
- `outer_candidate_metrics.csv`: inner-fold/run을 결합한 outer-selection 후보 점수, feasible 여부와 reliability rank.
- `outer_selections.csv`, `outer_selections.json`: 각 outer fold의 mean-CRPS 및 reliability 선택 결과.
- `final_candidate_fold_metrics.csv`: 전체 development selection의 inner-fold raw 후보 점수.
- `final_candidate_metrics.csv`: horizon별 12개 후보의 run-balanced CRPS/MAE/tail/exceedance/episode/rank.
- `final_selections.csv`, `final_selections.json`: holdout 평가 전에 동결한 최종 선택. 1초는 residual 1 s history/100 trees, 2초 reliability는 residual 5 s history/300 trees다.
- `outer_evaluation_tasks.csv`: outer test에서 수행된 fold/seed/condition/procedure task.
- `final_evaluation_tasks.csv`: reused-holdout 평가 및 최종 export task.

#### 예측과 평가

- `outer_predictions.parquet`: 신규 선택 절차의 grouped-outer target별 예측.
- `reused_holdout_predictions.parquet`: development refit 후 4개 holdout의 target별 예측.
- `per_target_predictions_all_models.parquet`: 신규 mean/reliability, 기존 full15/compact3, persistence를 공통 target으로 합친 canonical 대형 표.
- `aggregate_metrics.csv`: assessment/horizon/model/condition/representation/seed별 target-weighted와 run-balanced 지표.
- `per_run_metrics.csv`: run별 mean·tail·episode·persistence 비교.
- `per_cohort_metrics.csv`: cohort별 error distribution과 CRPS.
- `transition_metrics.csv`: stable/rising/falling별 오류·bias·tail.
- `interval_diagnostics.csv`: 전체 target의 80%/90% coverage, width, violations, interval score.
- `calibration_diagnostics.csv`: 15개 quantile 각각의 empirical coverage와 calibration error.
- `tolerance_exceedance.csv`: 50–1000 W threshold별 초과율.
- `failure_episodes.parquet`: 주 진단 threshold 500 W의 개별 연속 episode.
- `failure_episode_summary.csv`: 500 W episode의 count/평균/median/max 지속시간.
- `quantile_postprocessing_effects.csv`: raw q50와 full-grid monotone rearrangement q50의 변화 및 MAE.
- `improved_worsened_run_counts.csv`: persistence 대비 개선/악화/tie run 수.
- `paired_run_bootstrap_intervals.csv`: reliability-vs-mean, new-vs-old, metadata-vs-history의 2,000회 run bootstrap.
- `seed_sensitivity.json`: 17/42/2026 seed 예측 차이. 이 실행에서는 최대 q50 차이가 0 W다.
- `new_worst_cases.csv`: 신규 reliability 모델의 절대오차 상위 target.

#### Export·replay·보고

- `deployment_manifest.csv`: export bundle별 horizon, condition, candidate, source와 모델 수.
- `replay_predictions.parquet`: 저장한 신규 deployment bundle을 다시 불러 생성한 holdout replay 예측.
- `replay_verification.json`: batch/export prediction 동일성. full run의 최대 차이는 0 W다.
- `benchmark.json`: 두 horizon, 총 30 quantile model의 bundle 크기와 실제 latency.
- `report_en.md`: failure-aware full experiment의 최종 영문 보고서.
- `plots/tolerance_exceedance.png`: old/new/selection별 threshold 초과 곡선.
- `plots/transition_mae.png`: transition별 모델 MAE 꺾은선.
- `plots/interval_coverage.png`: nominal 대비 empirical interval coverage.
- `plots/candidate_tradeoff.png`: candidate CRPS와 지속오류 trade-off.
- `plots/historical_window_*.png`: 이전 audit worst case에서 old/new를 비교한 연속 창.
- 나머지 `plots/` 파일: 신규 모델 worst case 등 보고서용 진단 그림.

#### 대량 resume/cache 파일 패턴

- `checkpoints/candidate_<hash>.json`: 한 candidate-inner-fold 학습의 설정과 완료 metric.
- `checkpoints/evaluation_<hash>.json`: 한 selected evaluation task의 완료 record.
- `evaluation/<hash>/predictions.parquet`: outer fold에서 선택된 모델의 연속 test 예측.
- `evaluation/<hash>/full15_bundle.joblib`: 해당 fold/seed/condition의 15-quantile 모델.
- `final_evaluation/<hash>/predictions.parquet`: final development refit 모델의 holdout 예측.
- `final_evaluation/<hash>/full15_bundle.joblib`: final evaluation용 horizon별 full15 bundle.
- `training_samples/*.npz`: candidate/fold/horizon/history별 학습 row identity, label/weight와 sampling 결과. transition-weighted family의 train-only weight도 포함한다.
- 이 파일들은 hash가 task identity이므로 임의로 이름을 바꾸면 resume 연결이 끊어진다.

### Smoke 결과: `artifacts/smoke/`

Full과 같은 pipeline을 작은 실제 데이터 범위에서 검증한 결과다. 파일명과 역할은 `artifacts/full/`과 동일하지만 sample 수와 trial 수가 축소되어 있으며 연구 결론에는 사용하지 않는다.

## 신규 모델 번들

### Full: `improvements/failure_aware_20260921_v1/models/power-forecast-failure-aware-20260921-v1/`

- `full15_h1000_history_only.joblib`: 최종 1초 history-only 15-quantile 모델.
- `full15_h1000_history_meta.joblib`: 최종 1초 history+metadata 15-quantile 모델.
- `full15_h2000_history_only.joblib`: 최종 2초 history-only 15-quantile 모델.
- `full15_h2000_history_meta.joblib`: 최종 2초 history+metadata 15-quantile 모델.
- `history_only_full15_deployment.joblib`: 1초와 2초 history-only 모델을 묶은 30-model 배포 후보.
- `history_meta_full15_deployment.joblib`: 1초와 2초 metadata 모델을 묶은 30-model 주 배포 후보. 기존 모델을 자동 교체하지 않은 실험 번들이다.

### Smoke: `models/power-forecast-failure-aware-20260921-v1-smoke/`

작은 실제 데이터 smoke에서 export/reload/replay 경로를 확인하기 위한 bundle이다. 정확도 결론이나 실제 배포 후보로 사용하지 않는다.

## 신규 모델 이상치 audit

경로: `improvements/failure_aware_20260921_v1/validation/anomaly_audit_20260921_v1/`

이 audit는 최종 `reliability_selection + history_meta + full15 + seed 42`의 동결 예측 126,091개만 대상으로 한다. 재학습·튜닝·calibration을 하지 않았고 grouped-outer 12 runs와 이전에 관찰된 reused-holdout 4 runs를 분리한다.

- `validation_spec.json`: 대상 모델, error sign, transition, 50–1000 W threshold와 episode 정의.
- `data_integrity_checks.json`: 126,091 target, 16 unique runs, horizon identity, 중복, missing, monotone quantile 검사.
- `evaluation_inventory.csv`: assessment/run/cohort/segment/horizon/fold별 target 수, 시간 범위, gap과 held-out 상태.
- `aggregate_reconciliation.csv`: anomaly audit 재계산값과 full experiment target-weighted 집계의 일치 검사.
- `tail_error_metrics.csv`: assessment/horizon별 pooled 및 equal-run empirical p50/p90/p95/p99/max, bias와 worst-5%.
- `tolerance_exceedance.csv`: threshold별 전체/저예측/과대예측 초과율과 episode count/duration.
- `failure_episodes.parquet`: 모든 50/100/200/300/500/750/1000 W threshold의 개별 연속 episode.
- `failure_episode_summary.csv`: assessment/horizon/threshold별 episode count, median, p90, max, 총 지속시간.
- `outlier_targets_over_500w.parquet`: 절대오차가 500 W를 초과한 모든 신규 모델 target과 q10/q50/q90.
- `per_run_outlier_metrics.csv`: run별 tail, >500/>1000 W 비율, episode 수와 최장시간.
- `per_cohort_outlier_metrics.csv`: cohort별 동일 이상치 지표.
- `transition_metrics.csv`: stable/rising/falling별 MAE, bias, p95/p99/max와 초과율.
- `interval_diagnostics_by_transition.csv`: 각 transition의 nominal 80%/90% coverage, 위·아래 violation, width와 interval score.
- `quantile_calibration_by_transition.csv`: transition별 15개 quantile empirical calibration.
- `change_confusion_matrices.csv`: endpoint-change class의 count, precision, recall. falling/stable/rising 모두 직접 재계산한 수정 완료 표다.
- `worst_cases.csv`: assessment/horizon마다 최대 절대오차, 최대 저예측, 최대 과대예측, 최대 quantile rearrangement, 최장 500 W episode 시작점.
- `paired_uncertainty.csv`: 신규 reliability 모델과 mean/old/history-only의 fixed-prediction paired run bootstrap 구간.
- `source_artifact_hashes.csv`: 입력 canonical prediction, selection, spec와 metric 코드 hash.
- `execution_commands.txt`: 입력과 primary filter를 기록한 실행 메모.
- `execution.log`: audit 완료, test와 무결성 상태.
- `completion_status.json`: `completed`, integrity pass, target 수와 artifact 수.
- `report_en.md`: 이상치 audit의 영문 범위·핵심 tail/episode 결과와 해석 제한.
- `summary_ko.md`: 신규 모델 이상치 결과의 간단한 한국어 요약.
- `figures/absolute_error_ecdf.png`: assessment/horizon별 절대오차 분포.
- `figures/tolerance_exceedance.png`: 50–1000 W threshold 초과율.
- `figures/episode_duration_ecdf.png`: 500 W 연속 episode 지속시간 분포.
- `figures/coverage_by_transition.png`: transition별 nominal 80% interval coverage.
- `figures/timeseries_*_largest_abs.png`: assessment/horizon별 최대 절대오차 주변 10초 창.
- `figures/timeseries_*_longest_500w.png`: assessment/horizon별 최장 500 W episode와 앞뒤 3초 창.

## 비교 시각화: visualizations/error_comparison_20260921/

- README.md: 비교 대상과 그림별 설명.
- model_error_overview.png: horizon별 CRPS, MAE, p95 절대오차, 500 W 초과율 꺾은선.
- model_transition_errors.png: falling/rising/stable의 원본/개선 모델 MAE 꺾은선.
- model_error_quantiles.png: 절대오차 p50/p90/p95/p99 비교 곡선.
- model_tolerance_curves.png: 오차 threshold별 초과율 곡선.
- model_run_deltas.png: 각 run의 원본·개선 MAE를 연결한 선.
- model_transition_change_lines.png: horizon에 따른 transition별 MAE 개선율.
- model_per_run_paired.png: run별 원본/개선 MAE 산점도.
- predictability_crps.png, predictability_transition_crps.png: 별도 predictability 실험의 CRPS 곡선.
- predictability_history_gain_heatmap.png: history의 CRPSS 개선율 히트맵.
- model_comparison_summary.csv, predictability_summary.csv: 그림의 수치 요약.
- visualize_errors.py가 시각화를 생성하며 모델을 재학습하지 않는다. 기존/개선 모델 비교와 predictability 실험은 평가 모집단이 달라 별도로 표시한다.

## 시간별 오차 시각화: visualizations/error_over_time_20260922/

- README.md: 16개 실행별 평균 절대오차·P95 절대오차·부호오차 그래프로 가는 목록과 계산 정의.
- mae_over_time_overview.png: grouped-outer와 reused-holdout을 구분한 시간별 5초 run-balanced 평균 절대오차.
- runs/mae/*.png: 각 run의 200ms 시점별 절대오차와 5초 평균 절대오차.
- runs/p95/*.png: 각 run의 5초 구간 95번째 백분위 절대오차.
- runs/bias/*.png: 각 run의 200ms 시점별 부호오차와 5초 평균 부호오차.
- per_run_5s_error.csv: run/horizon/시간 구간별 target 수, 원본·개선 모델의 MAE, P95, 평균 부호오차.
- run_balanced_5s_mae.csv: 평가/horizon/시간 구간별 run-balanced 평균과 참여 run 수.
- visualize_error_timeline.py가 저장된 canonical 예측에서 그림을 다시 만든다. 가로축 target_time_s는 각 실행 안의 목표 시각(초)이다.

## 파일 형식 해석

- `.md`: 사람이 읽는 설명·진단·보고서.
- `.csv`: 작은/중간 크기의 구조화된 결과표. Excel보다 pandas 등에서 dtype을 명시해 읽는 것이 안전하다.
- `.parquet`: target별 예측이나 episode 같은 대형 열 지향 표. pandas/pyarrow로 필요한 열과 filter만 읽는 것이 좋다.
- `.json`: 설정, selection, manifest, 상태, 검증 결과.
- `.joblib`: Python/LightGBM 모델·encoder·schema를 포함한 reload 가능한 bundle. 신뢰할 수 있는 로컬 파일만 load해야 한다.
- `.npz`: NumPy 압축 배열. 학습 row identity, sample, weight 등 재현성 자료.
- `.png`: 보고서용 정적 진단 그림.
- `.pyc`, `__pycache__/`: Python 실행 cache. 삭제해도 원본 `.py`와 결과표에는 영향이 없으며 다시 생성될 수 있다.

## 결과를 비교할 때 주의할 점

- `grouped_outer`와 `reused_holdout`을 합쳐 하나의 일반화 점수로 해석하지 않는다.
- reused holdout은 이전 대화에서 이미 관찰된 데이터이므로 fresh prospective confirmation이 아니다.
- `run_balanced`와 `target_weighted`/`pooled_targets`는 값이 다를 수 있다. 어느 집계인지 반드시 확인한다.
- `history_only`와 `history_meta`는 metadata의 증분 가치를 비교하는 조건이다.
- `mean_crps_selection`과 `reliability_selection`은 동일 candidate table에서 선택 규칙만 다르다. 1초에서는 같은 모델을 선택했다.
- `full15`는 15 quantile 전체를 정렬한 표현이고 `compact3`는 q10/q50/q90만 정렬한 기존 export 표현이다.
- 50–1000 W threshold와 100 W transition 경계는 진단용이며 안전 한계가 아니다.
- 오류 episode 시간은 200 ms 간격 forecast error가 연속된 시간이지 물리적 제어 실패 시간은 아니다.
- maximum error는 관찰된 최댓값일 뿐 미래 상한이 아니다.
- `artifacts/full/`과 `audit_20260921_full_v3/`가 결론용이다. `smoke`, 이전 audit 버전, hash cache는 실행 검증·resume용이다.
