# Power Forecast Lite

실험 결과와 시간축 오차 그래프는 [RESULTS_INDEX.md](RESULTS_INDEX.md)에서 바로 찾을 수 있습니다. 생성 파일 전체의 용도는 [GENERATED_FILES_GUIDE.md](GENERATED_FILES_GUIDE.md)를 참조하세요.

CPU-only quantile forecasts for the sum of eight GPUs in one node. The two targets are the completed 200 ms power bins ending exactly 1,000 ms and 2,000 ms after the latest available completed bin. CPU/facility power is excluded.

The research evaluation uses 15 quantiles. Deployment exports q10/q50/q90 for each horizon, with row-wise sorting as the documented quantile-order correction. Static metadata are GPU model, workload type, documented AI model identifier, and batch size. A metadata identity change or a non-200 ms timestamp gap resets online state.

Commands (run from `/home/work/AIDC`):

```bash
pytest -q trace/model_200ms_1s_2s/tests
python trace/model_200ms_1s_2s/train.py --smoke
python trace/model_200ms_1s_2s/train.py
python trace/model_200ms_1s_2s/finalize.py
python trace/model_200ms_1s_2s/replay.py --package trace/model_200ms_1s_2s/models/power-forecast-lite-20260920/history_meta_deployment.joblib --output /tmp/replay.parquet
python trace/model_200ms_1s_2s/benchmark.py --package trace/model_200ms_1s_2s/models/power-forecast-lite-20260920/history_meta_deployment.joblib --output /tmp/benchmark.json
printf '%s\n' '{"timestamp_ms":0,"power_w":2000,"metadata":{"gpu":"H100","workload":"llm_training","batch":16,"ai_model":"8B"}}' | python trace/model_200ms_1s_2s/predict.py --package trace/model_200ms_1s_2s/models/power-forecast-lite-20260920/history_meta_deployment.joblib
```

Rerunning `train.py` resumes compatible atomic task checkpoints. A changed configuration is rejected by its signature. The package performs prediction only and contains no control or retraining path.
