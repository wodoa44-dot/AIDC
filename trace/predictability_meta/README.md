# Pooled metadata ablation

Run checks with `pytest tests -q`; a real-data smoke test with `python run.py --smoke`; and the main run with `python run.py --experiment-id main_20260915`.

The experiment is deliberately isolated from `trace/predictability`. It pools the prior 12 canonical high-resolution node traces and compares identical causal power histories against the same histories plus pre-issuance metadata. The reduced five-cell grid is fixed in configuration before scoring to make both protocols feasible on CPU.
