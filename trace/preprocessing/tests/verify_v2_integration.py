"""Executable integration verification; uses the six-dataset development fixture."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
import pandas as pd
from powerprep.main import main
from powerprep.discovery import stream_hash
from powerprep.contract import atomic_json

base=Path(__file__).parents[1].resolve()
out=base/'outputs/forecast_v2_complete_smoke_20260916'
args=['--config',str(base/'configs/forecast_v2.yaml'),'--output',str(out),'--limit','1']
assert json.loads((out/'progress.json').read_text())['status']=='completed'
files=list((out/'standardized').glob('*.parquet'))+list((out/'resolutions').glob('*/*.parquet'))
snapshot={str(p):(p.stat().st_mtime_ns,stream_hash(p)) for p in files}
for stage in ['preprocess','resolutions','windows','detect','report']:main([stage]+args)
assert all((Path(p).stat().st_mtime_ns,stream_hash(p))==value for p,value in snapshot.items())
windows=pd.read_parquet(out/'window_index.parquet')
assert windows.window_id.is_unique
assert (windows.feature_available_s<=windows.issue_s).all()
assert (windows.target_available_s>windows.issue_s).all()
for (run,resolution),group in windows.groupby(['run_id','resolution_id']):
    frame=pd.read_parquet(out/'resolutions'/resolution/f'{run}.parquet')
    for r in group.itertuples():
        support=frame.iloc[r.context_start_index:r.target_end_index+1]
        assert support.is_valid.all() and support.segment_id.nunique()==1
        assert support.segment_id.iloc[0]==r.segment_id
        assert frame.available_s.iloc[r.context_end_index]<=r.issue_s
        assert frame.available_s.iloc[r.target_start_index]>r.issue_s
assert len(pd.read_parquet(out/'events.parquet'))==len(pd.read_csv(out/'events.csv'))
atomic_json(out/'integration_verification.json',{'passed':True,'datasets':6,
             'valid_caches_preserved':len(files),'separate_stages_verified':['preprocess','resolutions','windows','detect','report'],
             'windows':len(windows),'event_export_complete':True,'models_trained':0})
print((out/'integration_verification.json').read_text())
