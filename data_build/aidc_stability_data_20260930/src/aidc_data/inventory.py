from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT.parents[1]
MANIFEST=ROOT/'trace/preprocessing/outputs/manifest.csv'


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def session_proxy(row)->str:
    if row.dataset=='highres_node':
        p=Path(row.source_path)
        return f'highres_sweep::{row.workload}::{row.hardware}::{p.parent.name}'
    if row.dataset=='nlr_training':
        return f'nlr_job::{getattr(row,"model", "unknown")}::{getattr(row,"node_count", "unknown")}'
    return f'{row.dataset}::source_group_unverified'


def audit(out:Path)->pd.DataFrame:
    out.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(MANIFEST)
    m['dataset_id']=m.dataset
    m['session_id']=m.apply(session_proxy,axis=1)
    m['split_group_id']=m.session_id
    m['independent_session_verified']=False
    m['evidence_type']='measured_existing'
    m['source_lineage']='trace/preprocessing/outputs/manifest.csv; raw source file; manifest source_hash not rehashed for every source'
    m['available_for_d1']=m.dataset.eq('highres_node') & ~m.is_duplicate
    m['available_for_d2']=m.available_for_d1
    m['exclusion_reason']=''
    m.loc[m.is_duplicate,'exclusion_reason']='exact source SHA256 alias'
    m.loc[m.dataset.eq('nlr_online_rate'),'exclusion_reason']='1ms aggregate grid uses future interpolation/backfill upstream; raw reconstruction requires separate track'
    m.loc[m.dataset.eq('nlr_training'),'exclusion_reason']='200ms published aggregate uses interpolation; raw asynchronous logs require separate reconstruction'
    m.to_csv(out/'inventory.csv',index=False)
    m[['dataset_id','source_path','source_hash','canonical_path','is_duplicate','run_id','session_id','split_group_id','source_lineage']].to_csv(out/'source_lineage.csv',index=False)
    cap=[
       ('GPU-only highres node power','available','raw GPU channels and timestamps','D1/D2','no verified sensor bandwidth/collection latency'),
       ('NLR published training/online aggregate','derivable','published aggregate and raw asynchronous logs','separate causal raw-log pilot','cannot use interpolated grid as online measurement'),
       ('GPU cap/filler command and service performance','requires_measurement','no paired command/performance trace in inventory','Phase0 collector/protocol','actual control effect and SLA cost'),
       ('PCC P/Q/V/f and ESS/UPS state','requires_measurement','no target-facility meter trace','D5 schema/collector','facility/grid claims'),
       ('single-area frequency model','derivable','published reduced-order parameter source','frequency-only scenario D4','no voltage/network/EMT or site validation'),
       ('dynamic network benchmark','requires_model','no installed ANDES case or verified target model','future D4','voltage/modes/weak-grid/UPS dynamics'),
    ]
    pd.DataFrame(cap,columns=['goal','availability','basis','possible_now','cannot_claim']).to_csv(out/'capability_matrix.csv',index=False)
    requirements=[
       ('D3','paired cap/filler commands, enforced limits, GPU/node power, step or request latency','blocked_hardware','specific authorized GPU UUID and workload; instrumented trial'),
       ('D4 network','topology, base, dynamic generator/load/converter parameters, operating point, license','blocked_model','validated benchmark package/case'),
       ('D5','PCC P/Q/V/f, ESS SoC, UPS mode, synchronized timestamps','blocked_source','facility meters and permission'),
       ('metadata dynamic','phase, queue, arrival, plan publication and available_at','blocked_source','instrumented workload log'),
    ]
    pd.DataFrame(requirements,columns=['data_kind','required','status','next_input']).to_csv(out/'missing_data_requirements.csv',index=False)
    return m
