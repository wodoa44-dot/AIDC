"""Integration assertions against a completed real-data run (opt-in)."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from src.core import crps


def test_real_predictions():
    directory=os.environ.get('PREDICTABILITY_CHECK_OUTPUT')
    if not directory:pytest.skip('Set PREDICTABILITY_CHECK_OUTPUT to completed experiment path')
    out=Path(directory);manifest=pd.read_csv(out/'input_manifest.csv');metrics=pd.read_csv(out/'metrics.csv')
    assert manifest.source_hash.is_unique
    for file in (out/'predictions').glob('*.parquet'):
        p=pd.read_parquet(file);columns=[c for c in p if c.startswith('q_')];q=np.array([float(c[2:]) for c in columns])
        keys=['run_id','segment_id','target_bin']
        reference=None
        for condition,g in p.groupby('condition'):
            assert not g.duplicated(keys).any()
            assert (g.split==2).all()
            np.testing.assert_allclose(g.target_s-g.issue_s,g.horizon_ms/1000)
            assert np.isfinite(g[columns]).all().all()
            assert (np.diff(g[columns],axis=1)>=0).all()
            np.testing.assert_allclose(crps(g.y,g[columns].to_numpy(),q),g.crps)
            ids=g[keys+['y','issue_s','target_s']].reset_index(drop=True)
            if reference is None:reference=ids
            else:pd.testing.assert_frame_equal(reference,ids)
            m=metrics[(metrics.cohort==g.cohort.iloc[0])&(metrics.resolution_ms==g.resolution_ms.iloc[0])&(metrics.horizon_ms==g.horizon_ms.iloc[0])&(metrics.condition==condition)]
            assert len(m)==1
            np.testing.assert_allclose(m.mean_crps_w.iloc[0],g.crps.mean())
        for run,g in p.groupby('run_id'):
            cutoff=.8*float(manifest.set_index('run_id').loc[run,'duration_s'])
            assert (g.issue_s>=cutoff).all()
    assert len(list((out/'predictions').glob('*.parquet')))==len(metrics)//3
