"""Evidence-led grouping; recording/session independence is never inferred from filename."""
import json
from pathlib import Path
import pandas as pd

DATA=Path('/home/work/AIDC/trace/dataset/High-resolution-AI-Data-Center-Training-Workloads-Dataset/High-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare')

def verify_groups(cfg,out,sha):
    inventory=pd.read_csv(Path(cfg['source'])/'inventory.csv')
    inventory=inventory[inventory.dataset.eq('highres_node')].copy()
    rows=[]; files=[]
    for run,g in inventory.groupby('run_id',sort=True):
        canonical=g.loc[~g.is_duplicate.astype(str).str.lower().eq('true')].iloc[0]
        source=Path(canonical.source_path)
        assert sha(source)==canonical.source_hash,('raw source changed',str(source))
        frame=pd.read_csv(source,usecols=['timestamp'])
        t=pd.to_datetime(frame.timestamp,errors='coerce'); assert t.notna().all() and t.is_monotonic_increasing
        start=t.min(); end=t.max()
        assert start.date()==end.date(),'Multi-day record needs overlap-component grouping review'
        header=pd.read_csv(source,nrows=0).columns.tolist()
        identifiers=[x for x in header if any(y in x.lower() for y in ['session','experiment_id','job_id','run_id'])]
        for alias in g.source_path:
            path=Path(alias); digest=sha(path); assert digest==canonical.source_hash
            files.append({'run_id':run,'source_file':str(path),'sha256':digest,'exact_duplicate_alias':str(path)!=str(source)})
        rows.append({'run_id':run,'source_file':str(source),'source_sha256':canonical.source_hash,
                     'collection_session_id':'inferred-recording-'+str(canonical.source_hash)[:16],
                     'independent_group_id':'','assumed_group_id':'day-'+start.strftime('%Y-%m-%d'),
                     'collection_day':start.strftime('%Y-%m-%d'),'first_timestamp':start.isoformat(),'last_timestamp':end.isoformat(),
                     'timestamp_count':len(t),'explicit_session_columns':json.dumps(identifiers),
                     'basis':'README describes 32 node training sessions; raw timestamps identify non-overlapping recordings. Same recorded calendar day conservatively clustered; not hardware/model/batch grouping.',
                     'verification_status':'recording_interval_verified; independent_session_unverified',
                     'uncertainty':'No explicit session/job identifier or reset/randomization log; timezone unspecified. Different dates also not proven statistically independent. Date confounded with hardware/workload.',
                     'aliases':json.dumps(g.source_path.tolist())})
    result=pd.DataFrame(rows); overlaps=[]
    for i,a in result.iterrows():
        for _,b in result.iloc[i+1:].iterrows():
            if max(a.first_timestamp,b.first_timestamp)<=min(a.last_timestamp,b.last_timestamp):
                overlaps.append({'run_a':a.run_id,'run_b':b.run_id,'same_assumed_group':a.assumed_group_id==b.assumed_group_id})
                assert a.assumed_group_id==b.assumed_group_id,'Cross-day overlap requires merged groups'
    result.to_csv(out/'group_manifest.csv',index=False)
    pd.DataFrame(files).to_csv(out/'source_aliases.csv',index=False)
    pd.DataFrame(overlaps,columns=['run_a','run_b','same_assumed_group']).to_csv(out/'overlap_checks.csv',index=False)
    mapping=result.set_index('run_id').assumed_group_id.to_dict()
    old=json.loads((Path(cfg['source'])/'split_manifest.json').read_text())['folds']; impacted=[]
    for s in old:
        overlap=sorted({mapping[r] for r in s['train_runs']}&{mapping[r] for r in s['test_runs']})
        impacted.append({'old_fold':s['fold'],'shared_assumed_collection_groups':overlap,'regenerate_under_day_grouping':bool(overlap)})
    docs=[]
    for p in sorted(DATA.rglob('*')):
        if p.is_file() and (p.suffix.lower() in ('.md','.json','.yaml','.yml','.log','.txt') or any(x in p.name.lower() for x in ('readme','metadata','session'))):
            docs.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    evidence={'canonical_runs':len(result),'raw_files_including_aliases':len(files),'exact_duplicate_aliases':len(files)-len(result),
              'overlapping_pairs':len(overlaps),'assumed_collection_days':result.assumed_group_id.nunique(),'verified_independent_groups':0,
              'readme':str(DATA/'README.md'),'session_document_search':docs,'old_split_impact':impacted,
              'collection_group_verification':'blocked_independence_evidence','independent_group_statistics':'blocked_independence_evidence'}
    (out/'group_evidence.json').write_text(json.dumps(evidence,indent=2))
    (out/'group_audit_ko.md').write_text('# 수집 group 검증\n\n원본 README는 32개 node training session을 설명한다. CSV timestamp와 SHA256를 직접 재확인했다. 파일명이나 장치 설정으로 독립성을 확정하지 않았다.\n\n'+result[['run_id','collection_day','first_timestamp','last_timestamp']].to_markdown(index=False)+'\n\n42개 경로의 32개 고유 기록: 동일 SHA256 별칭은 같은 run으로 통합. 명시 session/job ID, session reset 및 무작위화 로그는 발견되지 않았다. 독립 group ID는 빈 값으로 보존한다.\n\n수집일 4개를 보수적 가정 group으로 사용한다. 날짜와 hardware/workload가 교락되어 날짜 holdout 효과와 조합 일반화 효과는 분리할 수 없다. 기존 outer 5개 split 모두 같은 날짜를 train/test에 공유하므로 이번 보수적 가정 아래 4개 leave-day-out split으로 교체한다. 예전 결과는 삭제하지 않는다. 날짜 간 독립성도 미확인으로, day CI는 가정 기반이며 정식 독립-group CI는 blocked이다.\n')
    return result
