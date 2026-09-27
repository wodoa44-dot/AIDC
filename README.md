# AIDC forecasting research code

AIDC 전력 수요 예측, metadata 평가, event 감지와 검증용 연구 코드입니다. 이 게시본은 실행 코드, 설정과 테스트만 포함합니다. 원본 데이터, 파생 결과, 모델, 그림과 실행 로그는 포함하지 않습니다.

각 실험을 재현하려면 해당 스크립트가 지정하는 데이터 경로에 원본 데이터를 별도로 준비해야 합니다. 데이터 없이 실행 가능한 핵심 테스트:

```bash
PYTHONPATH=trace/grid_stability_forecast python -m unittest discover -s trace/grid_stability_forecast/tests -v
```

주요 실험 진입점은 `trace/grid_stability_forecast/`의 `run_nested_cv.py`, `run_nested_horizon.py`, `run_nlr_training_cv.py`, `train_onset_aligned.py`입니다. 실제 장치나 계통 제어 명령은 이 코드에 포함되지 않습니다.
