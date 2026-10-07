# 예측 모델 실험

`aidc_grid_metadata_20260929/`는 history 및 metadata 기반 예측 모델의 주 실험이다. `src/aidc_exp/training.py`, `features.py`, `targets.py`, `evaluation.py`에서 학습·입력·target·평가를 확인할 수 있다.

`nested_predictor_cache/`는 제어 실험에서 사용한 최신 outer 및 inner 예측 cache다. `diagnosis_endpoint_predictions/`와 `diagnosis_trajectory_predictions/`는 최종 진단의 예측 산출물이다. 전체 진단의 제어 코드와 보고서는 [제어 모델 폴더](../control_models/aidc_forecast_diagnosis_20261001/)에 있다.

모든 실험 항목은 원본에 대한 심볼릭 링크이며 실행·재개는 원본 README의 경로를 따른다.
