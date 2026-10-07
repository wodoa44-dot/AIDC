"""Verify all 32 raw source hashes and reconstruct prior coverage/aggregation."""
import importlib.util
import numpy as np
import pandas as pd
from common import PROJECT, digest
from data import PREPARED, GROUPS


def audit_sources(out):
    path = (
        PROJECT
        / "experiments/aidc_grid_metadata_20260929/src/aidc_exp/preprocessing.py"
    )
    s = importlib.util.spec_from_file_location("source_preprocessing", path)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    raw = pd.read_parquet(PREPARED, columns=["run_id", "bin", "segment", "current_w"])
    groups = pd.read_csv(GROUPS)
    rows = []
    for _, g in groups.iterrows():
        source_hash = digest(g.source_file)
        assert source_hash == g.source_sha256, f"Raw input changed: {g.run_id}"
        bins = module.load_highres(g.source_file)
        cached = raw[raw.run_id.eq(g.run_id)].sort_values("bin")
        assert np.array_equal(bins.bin, cached.bin)
        assert np.array_equal(bins.segment, cached.segment)
        np.testing.assert_allclose(
            bins.power_w, cached.current_w, rtol=1e-12, atol=1e-8, equal_nan=True
        )
        rows.append(
            {
                "run_id": g.run_id,
                "source_sha256": source_hash,
                "hash_matches": True,
                "raw_bin_reconstruction_matches": True,
                "bins": len(bins),
                "invalid_bins": int((~bins.valid).sum()),
                "observed_dt_s": float(bins.observed_dt_s.iloc[0]),
                "coverage_rule": "0.8; gap_factor5; no interpolation",
                "preprocessing_sha256": digest(path),
            }
        )
    pd.DataFrame(rows).to_csv(out / "raw_source_validation.csv", index=False)
