"""Online inference contract for full-15 absolute or residual bundles."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE))
from src.core import power_features, raw_predict, rearrange, transform_one


def predict_item(item: dict, power_feature_row: np.ndarray, metadata: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Return raw power quantiles, ordered power quantiles, and crossing rate."""
    x = transform_one(power_feature_row, metadata, item["encoder"])
    raw = raw_predict(item["models"], x)
    if item.get("family", item.get("parameters", {}).get("family", "absolute")) != "absolute":
        raw = raw + float(power_feature_row[0])
    ordered, crossing = rearrange(raw)
    return raw, ordered, crossing


class ForecastService:
    def __init__(self, package_path, response_quantiles=(.1, .5, .9)):
        self.package_path = Path(package_path)
        self.bundle = joblib.load(self.package_path)
        self.resolution_ms = int(self.bundle["resolution_ms"])
        self.response_quantiles = tuple(map(float, response_quantiles))
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
            self.reset()
        if self._last_timestamp_ms is not None and timestamp_ms - self._last_timestamp_ms != self.resolution_ms:
            self.reset()
        self._metadata = identity; self._last_timestamp_ms = timestamp_ms
        if not np.isfinite(power_w) or power_w < 0:
            self.reset(); return {"status": "unavailable", "reason": "invalid_power"}
        self._values.append(float(power_w))
        needed = self.max_history_ms // self.resolution_ms
        if len(self._values) < needed:
            return {"status": "not_ready", "reason": "insufficient_history",
                    "have_bins": len(self._values), "need_bins": needed}
        md = dict(zip(("gpu", "workload", "batch", "ai_model"), identity))
        forecasts = []
        for horizon, item in sorted(self.bundle["horizons"].items(), key=lambda pair: int(pair[0])):
            pf = power_features(self._values, self.resolution_ms, int(item["history_ms"]))
            raw, ordered, crossing = predict_item(item, pf, md)
            all_quantiles = dict(zip(map(float, item["quantiles"]), ordered[0]))
            requested = {f"q{int(round(q*1000)):03d}": float(all_quantiles[q]) for q in self.response_quantiles}
            forecasts.append({"issue_time_ms": timestamp_ms, "target_time_ms": timestamp_ms + int(horizon),
                "horizon_ms": int(horizon), "quantiles": requested,
                "full_quantile_count_computed": len(all_quantiles),
                "raw_q50_w": float(raw[0, list(map(float, item["quantiles"])).index(.5)]),
                "crossing_corrected": bool(crossing), "units": "W",
                "model_version": self.bundle["model_version"]})
        return {"status": "ready", "forecasts": forecasts}
