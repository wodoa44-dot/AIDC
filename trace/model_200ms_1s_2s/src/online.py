"""Stateful prediction API for timestamped completed 200 ms bins."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import joblib
import numpy as np

from .core import power_features, raw_predict, rearrange, transform_one


class ForecastService:
    def __init__(self, package_path):
        self.package_path = Path(package_path)
        self.bundle = joblib.load(self.package_path)
        self.resolution_ms = int(self.bundle["resolution_ms"])
        self.max_history_ms = max(int(x["history_ms"]) for x in self.bundle["horizons"].values())
        self._values = deque(maxlen=self.max_history_ms // self.resolution_ms)
        self._last_timestamp_ms = None
        self._metadata = None

    def reset(self):
        self._values.clear(); self._last_timestamp_ms = None; self._metadata = None

    def update(self, timestamp_ms: int, power_w: float, metadata: dict):
        timestamp_ms = int(timestamp_ms)
        identity = tuple(metadata.get(k) for k in ("gpu", "workload", "batch", "ai_model"))
        if self._metadata is not None and identity != self._metadata:
            self.reset()  # metadata is effective at this timestamp; segment identity changed
        if self._last_timestamp_ms is not None and timestamp_ms - self._last_timestamp_ms != self.resolution_ms:
            self.reset()
        self._metadata = identity; self._last_timestamp_ms = timestamp_ms
        if not np.isfinite(power_w) or power_w < 0:
            self.reset(); return {"status": "unavailable", "reason": "invalid_power"}
        self._values.append(float(power_w))
        if len(self._values) < self.max_history_ms // self.resolution_ms:
            return {"status": "not_ready", "reason": "insufficient_history",
                    "have_bins": len(self._values), "need_bins": self.max_history_ms // self.resolution_ms}
        md = dict(zip(("gpu", "workload", "batch", "ai_model"), identity))
        forecasts = []
        for horizon, item in sorted(self.bundle["horizons"].items(), key=lambda x: int(x[0])):
            pf = power_features(self._values, self.resolution_ms, int(item["history_ms"]))
            x = transform_one(pf, md, item["encoder"])
            ordered, crossing = rearrange(raw_predict(item["models"], x))
            forecasts.append({"issue_time_ms": timestamp_ms, "target_time_ms": timestamp_ms + int(horizon),
                "horizon_ms": int(horizon), "quantiles": {f"q{int(q*100):02d}": float(v) for q, v in zip(item["quantiles"], ordered[0])},
                "crossing_corrected": bool(crossing), "units": "W", "model_version": self.bundle["model_version"]})
        return {"status": "ready", "forecasts": forecasts}
