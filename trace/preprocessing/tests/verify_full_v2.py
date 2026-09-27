"""Wait for the complete data run, validate persisted lineage, and produce Korean handoff."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]))
from powerprep.contract import atomic_json
from powerprep.discovery import stream_hash
from powerprep.forecast import bin_splits
base=Path(__file__).parents[1].resolve()
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=base/'outputs/forecast_v2_1_20260916')
parser.add_argument('--log',type=Path,default=Path('/tmp/powerprep_v2_1.log'))
args=parser.parse_args()
out=args.output.resolve()
while not (out/'progress.json').exists(): time.sleep(1)
while True:
    state=json.loads((out/'progress.json').read_text())
    if state['status']=='failed':raise RuntimeError((out/'failure.json').read_text())
    if state['stage']=='all' and state['status']=='completed':break
    pid=json.loads((out/'provenance.json').read_text())['pid']
    if not Path(f'/proc/{pid}').exists():raise RuntimeError('Preprocessing process disappeared before completion')
    print(json.dumps(state),flush=True)
    time.sleep(30)
manifest=pd.read_csv(out/'manifest.csv');canonical=manifest[~manifest.is_duplicate]
metadata=pd.read_parquet(out/'run_metadata.parquet')
assert metadata.run_id.is_unique and len(metadata)==len(canonical)
q=pd.read_csv(out/'native_quality.csv');assert len(q)==len(canonical)
assert (q.native_samples==q.valid_samples+q.excluded_samples).all()
segments=pd.read_parquet(out/'segments.parquet')
actual=segments.groupby('run_id').duration_s.sum().reindex(q.run_id,fill_value=0).to_numpy()
np.testing.assert_allclose(actual,q.valid_duration_s,atol=1e-8,rtol=1e-10)
verified=0
for side in (out/'cache').glob('*.json'):
    record=json.loads(side.read_text())
    for path,checksum in record['files'].items():assert stream_hash(path)==checksum,path
    verified+=1
splits=pd.read_csv(out/'split_manifest.csv')
supported=splits[(splits.support_status=='supported')&(splits.protocol=='held_out_group')]
assert (supported.groupby('group_id').split.nunique()<=1).all()
windows=pd.read_parquet(out/'window_index.parquet');assert windows.window_id.is_unique
frames_checked=0
for (run,resolution),group in windows.groupby(['run_id','resolution_id']):
    frame=pd.read_parquet(out/'resolutions'/resolution/f'{run}.parquet')
    assert not frame.duplicated(['run_id','segment_id','resolution_id','bin_index']).any()
    first=group.context_start_index.to_numpy(int);last=group.target_end_index.to_numpy(int)
    assert (first>=0).all() and (last<len(frame)).all()
    assert (frame.segment_id.to_numpy()[first]==frame.segment_id.to_numpy()[last]).all()
    bad=np.r_[0,np.cumsum(~frame.is_valid.to_numpy())]
    assert ((bad[last+1]-bad[first])==0).all()
    t=frame.available_s.to_numpy()
    assert np.all(t[group.context_end_index.to_numpy(int)]<=group.issue_s)
    assert np.all(t[group.target_start_index.to_numpy(int)]>group.issue_s)
    for protocol,g in group.groupby('protocol'):
        a=next(splits[(splits.run_id==run)&(splits.protocol==protocol)].itertuples(index=False))
        labels=bin_splits(frame,a)
        for split,z in g.groupby('split'):
            bad=np.r_[0,np.cumsum(labels!=split)]
            assert ((bad[z.target_end_index.to_numpy(int)+1]-bad[z.context_start_index.to_numpy(int)])==0).all()
    frames_checked+=1
# Independently reconstruct unique training-bin statistics, not window-weighted values.
norm=json.loads((out/'normalization.json').read_text())
for key,expected in norm['parameters'].items():
    protocol,rid,dataset=key.split('/')
    if expected['status']!='fitted':continue
    n=0;total=0.;squares=0.
    sub=splits[(splits.protocol==protocol)&(splits.dataset_id==dataset)&(splits.support_status=='supported')]
    for a in sub.itertuples(index=False):
        path=out/'resolutions'/rid/f'{a.run_id}.parquet'
        if not path.exists():continue
        f=pd.read_parquet(path)
        x=f.loc[(bin_splits(f,a)=='train')&f.is_valid,'power_w'].to_numpy()
        n+=len(x);total+=float(x.sum());squares+=float(np.dot(x,x))
    mean=total/n;scale=np.sqrt(max(0,squares/n-mean*mean))
    assert n==expected['training_observations']
    np.testing.assert_allclose([mean,scale or 1.],[expected['mean_w'],expected['scale_w']],atol=1e-7,rtol=1e-9)
result={'passed':True,'source_files':len(manifest),'canonical_runs':len(canonical),'aliases':int(manifest.is_duplicate.sum()),
        'datasets':len(q.dataset_id.unique()),'native_samples':int(q.native_samples.sum()),
        'prepared_samples':int(q.valid_samples.sum()),'windows':len(windows),'checked_window_files':frames_checked,
        'verified_cache_records':verified,'normalization_training_only_verified':True,'models_trained':0,
        'completed_utc':time.strftime('%FT%TZ',time.gmtime())}
atomic_json(out/'full_verification.json',result)
summary=pd.read_csv(out/'dataset_summary.csv');summary['valid_hours']=summary.valid_duration_s/3600
counts=pd.read_csv(out/'window_counts.csv').groupby(['dataset_id','protocol','split']).windows.sum().reset_index()
unknown=metadata[['precision','batch_semantics','gradient_accumulation','source_sensor_cadence_s','sensor_averaging_window_s','power_cap_w']].isna().sum().rename('unknown_runs').to_frame()
text=['# 전처리 v2 완료 인계','',f'전체 {len(manifest):,}개 파일을 조사하고 바이트 동일 별칭 {int(manifest.is_duplicate.sum()):,}개를 구분해 {len(canonical):,}개 canonical run을 처리했습니다.',
      '기존 예측 실험과 과거 출력은 변경하지 않았으며 새 예측 모델은 학습하지 않았습니다.','',
      '## 데이터별 실제 처리 결과','',summary.to_markdown(index=False),'',
      '표준화 원본은 유효하지 않은 관측도 보존하며 prepared는 유효 관측만 제공합니다. 유효 시간은 구간별 마지막 관측-첫 관측의 합이며 공백과 추정 꼬리 시간을 포함하지 않습니다.',
      '','## 변경 및 근거','',
      '원본 순서 검증을 집계보다 먼저 적용하고 구간 ID, 행 계보, 품질 비트, GPU 채널을 보존했습니다. 대문자 .CSV 단일 GPU 자료도 포함했습니다. '
      '원본 메타데이터 행·파일명·README·설정·모델 크기 설명을 보존하며 검증/선언/추론/미상 상태를 분리합니다. '
      '원본 설명의 모델 크기 충돌은 확정하지 않습니다. 누락값을 기본 모델명·노드 수로 사실화하지 않습니다.',
      '',unknown.to_markdown(),'',
      '## split/window 수','',counts.to_markdown(index=False),'',
      '1초: context 10초/horizon 5초/stride 5초, 0.2초: context 10초/horizon 2초/stride 2초. '
      'output interval은 각 해상도와 동일하며 미래 궤적 인덱스만 저장합니다. train/validation/test 경계와 구간 경계를 넘지 않습니다. '
      '공유 원본 로그를 사용하는 NLR 추론 데이터는 보수적 그룹화 때문에 split 설정이 지원 불가일 수 있으며 forecast_support.csv에 이유를 기록합니다.',
      '','단일 GPU 일부 파일의 분:초 표기는 절대 시각의 시(hour)를 복원하지 않고 구간 내부 상대 시간만 복구했습니다. 되감김은 별도 epoch로 분할하며 hour_unknown_valid_samples에 기록합니다.',
      '','## 해상도와 한계','',
      '원본을 보존하고 0.1/0.2/1초 중 원본 기록 간격보다 거칠거나 같은 변형만 생성했습니다. availability는 bin 오른쪽 끝입니다. '
      'NLR 보간 및 일부 후방 채움은 인과적으로 검증되지 않았고 격자 coverage는 독립 센서 coverage가 아닙니다. '
      '전력 범위가 다른 GPU-only 단일 GPU/8 GPU 합계와 CPU+GPU 전체 작업은 분리합니다.',
      '','## 검증 및 선택적 이벤트','',
      '21개 회귀 테스트와 6개 데이터셋 통합 검증이 통과했습니다. 최종 산출물 체크섬, 전체 window 경계/가용 시점, '
      'alias 그룹 격리, 훈련 bin 전용 정규화 통계를 별도로 검증했습니다. 이벤트 v2는 최종 회복 구간의 방향성 교차를 사용하며 '
      '미해결·검열·동일 샘플 교차에 회복 시간 0을 부여하지 않습니다. 기울기는 임계값 이후로 명명했습니다. '
      '이벤트는 소규모 통합에서만 실행했으며 전체 처리에서는 선택 기능으로 비활성화했습니다.',
      '','## 재현','',
      '```bash\ncd /home/work/AIDC\nPYTHONPATH=trace/preprocessing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python -m powerprep all --config trace/preprocessing/configs/forecast_v2.yaml --output trace/preprocessing/outputs/forecast_v2_1_20260916\n```',
      '', '출력: `trace/preprocessing/outputs/forecast_v2_1_20260916/`. 상세 계약은 data_dictionary.md, 검사 증거는 full_verification.json, 독립 단계 명령은 ../../FORECAST_V2.md에 있습니다.',
      '예측 성능이나 제어 가능성을 주장하지 않습니다.']
(out/'handoff_ko.md').write_text('\n'.join(text)+'\n')
shutil.copyfile(args.log,out/'execution.log')
print(json.dumps(result),flush=True)
