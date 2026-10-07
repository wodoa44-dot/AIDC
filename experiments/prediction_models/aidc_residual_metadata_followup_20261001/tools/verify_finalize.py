"""Read-only independent checks, then deterministic report supplement; no fits."""
import sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import markdown
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data import *
out=Path(sys.argv[1]).resolve();pilot=PILOT/'outputs/pilot_v1';start=time.perf_counter()
# Confirm original cache/manifest and saved model checksums against pilot checkpoints.
for n,h in json.loads((pilot/'sampling_checkpoint.json').read_text()).items():assert digest(pilot/n)==h
for file in pilot.glob('fits/*/complete.json'):
 r=json.loads(file.read_text())
 for n,h in r['files'].items():assert digest(file.parent/n)==h
nested=pd.read_parquet(out/'nested_train_anchors.parquet');ev=pd.read_parquet(out/'evaluation_anchors.parquet')
original=pd.read_parquet(pilot/'evaluation_anchors.parquet');original=original[original.collection_day.isin(ev.collection_day.unique())]
np.testing.assert_array_equal(ev[KEY].to_numpy(),original[KEY].to_numpy())
oldtr=pd.read_parquet(pilot/'train_anchors.parquet')
for run,g in nested.groupby('run_id'):
 np.testing.assert_array_equal(g[g.cap==300][KEY].to_numpy(),oldtr[oldtr.run_id==run][KEY].to_numpy())
 keys=[set(map(tuple,g[g.cap==c][KEY].to_numpy())) for c in [100,300,600]];assert keys[0]<=keys[1]<=keys[2]
splits=json.loads((out/'split_manifest.json').read_text())['folds'];registry=pd.read_csv(out/'fit_registry.csv');assert len(registry)==27
inference_seconds=0.
for dest in sorted((out/'fits').iterdir()):
 r=json.loads((dest/'complete.json').read_text());ident=r['identity'];cap=ident['cap'];fold=ident['fold'];model=ident['model']
 for n,h in r['files'].items():assert digest(dest/n)==h
 tr=nested[(nested.cap==cap)&nested.collection_day.isin(splits[fold]['train_days'])];va=ev[ev.collection_day.isin(splits[fold]['validation_days'])]
 x,xv,maps,_=encode(tr,va,r['feature_columns']);assert maps==r['encoding']
 assert frame_hash(x)==ident['feature_hash'] and frame_hash(tr[KEY])==ident['train_anchor_hash']
 stored=pd.read_parquet(dest/'weights.parquet');np.testing.assert_allclose(weights(tr),stored.sample_weight)
 assert frame_hash(stored[KEY])==frame_hash(tr[KEY])
 assert not set(tr.run_id)&set(va.run_id)
 t=time.perf_counter();p=lgb.Booster(model_file=str(dest/'q50.txt')).predict(xv,num_threads=4);inference_seconds+=time.perf_counter()-t
 if model!='A':p+=va.current_w.to_numpy()
 np.testing.assert_allclose(p,pd.read_parquet(dest/'predictions.parquet').prediction_w,atol=1e-8)
# Check cap300 repeats prior pilot results, including ordinal encoding and weighting.
pred=pd.read_parquet(out/'predictions.parquet');old=pd.read_parquet(pilot/'pilot_predictions.parquet')
old=old[(old.scope=='development')&(old.region=='whole')&(old.postprocess=='raw')]
cap300=[]
for model,name in [('A','absolute_h5'),('R','residual_h5'),('RM','residual_h5_metadata')]:
 a=pred[(pred.cap==300)&(pred.model==model)].sort_values(KEY);b=old[old.model==name].sort_values(KEY)
 np.testing.assert_array_equal(a[KEY],b[KEY]);np.testing.assert_allclose(a.prediction_w,b.prediction_w,atol=1e-8)
 cap300.append({'model':model,'max_prediction_difference_w':float(abs(a.prediction_w.to_numpy()-b.prediction_w.to_numpy()).max())})
pd.DataFrame(cap300).to_csv(out/'pilot_300_reproduction.csv',index=False)
# All normal/rising/falling contributions and magnitude summaries of saved predictions.
prior=pd.read_parquet(out/'prior_diagnostic_predictions.parquet');details=[]
for (scope,fold,model),g in prior.groupby(['scope','fold','model']):
 r=json.loads((pilot/f'fits/{scope}_{fold}_{model}/complete.json').read_text());q50=r['threshold_q50_w'];q90=r['threshold_q90_w']
 masks={'normal':abs(g.delta)<=q50,'rising':g.delta>q50,'falling':g.delta<-q50,'large_change':abs(g.delta)>q90,'0-q50':abs(g.delta)<=q50,'q50-q90':(abs(g.delta)>q50)&(abs(g.delta)<=q90),'>q90':abs(g.delta)>q90}
 for region,mask in masks.items():
  for run,z in g[mask].groupby('run_id'):
   details.append(dict(scope=scope,analysis_label=z.analysis_label.iloc[0],fold=fold,model=model,run_id=run,collection_day=z.collection_day.iloc[0],hardware=z.hardware.iloc[0],workload=z.workload.iloc[0],region=region,n=len(z),mae=z.abs_error.mean(),gain=z.gain.mean(),mean_abs_delta_hat=abs(z.delta_hat).mean(),median_abs_delta_hat=abs(z.delta_hat).median()))
d=pd.DataFrame(details);d.to_csv(out/'prior_magnitude_diagnostics_per_run.csv',index=False)
d.groupby(['analysis_label','model','region'])[['mae','gain','mean_abs_delta_hat','median_abs_delta_hat']].mean().reset_index().to_csv(out/'prior_magnitude_diagnostics_summary.csv',index=False)
# Confirm all 1:1 PNG/SVG/CSV artifacts and their numeric provenance.
for p in (out/'figures').glob('*.png'):
 assert p.stat().st_size>0 and p.with_suffix('.svg').exists() and (out/'plotting_data'/(p.stem+'.csv')).exists()
for plot,table in [('learning_mae','metrics_summary'),('learning_p95','metrics_summary'),('learning_by_day','metrics_by_day'),('gain_contribution','gain_decomposition')]:
 pd.testing.assert_frame_equal(pd.read_csv(out/'plotting_data'/(plot+'.csv')),pd.read_csv(out/(table+'.csv')))
verify=json.loads((out/'verification.json').read_text());verify.update(pilot_cache_checksums_verified=True,all_pilot_fit_files_verified=True,all_followup_fit_files_verified=True,all_saved_predictions_reproduced_by_loaded_models=True,cap300_pilot_predictions_reproduced=True,source_identity_postrun_verified=True,plot_numeric_provenance_verified=True,validation_fits=0,verification_inference_seconds=inference_seconds)
identity=json.loads((out/'identity.json').read_text())
for p,h in identity['inputs'].items():assert digest(p)==h
for p,h in identity['code'].items():assert digest(p)==h
atomic_json(out/'verification.json',verify)
# Deterministic postprocessing after pipeline report-only; this separate script never trains.
s=pd.read_csv(out/'metrics_summary.csv');days=pd.read_csv(out/'metrics_by_day.csv');reg=pd.read_csv(out/'region_metrics.csv');de=pd.read_csv(out/'gain_decomposition.csv');sample=pd.read_csv(out/'sampling_summary.csv')
summary=s.pivot(index='cap',columns='model',values='mae').reset_index();summary['A_minus_R']=summary.A-summary.R;summary['R_minus_RM']=summary.R-summary.RM
big=reg[(reg.model=='R')&(reg.region=='large_change')][['cap','mae','gain']]
normal=de[(de.model=='R')&(de.region=='normal')][['cap','gain_contribution']]
meta=s[s.model.isin(['R','RM'])].pivot(index='cap',columns='model',values=['mae','p95_abs_error','p99_abs_error']);effect=pd.DataFrame({'cap':meta.index,'mae_gain':meta['mae']['R']-meta['mae']['RM'],'p95_gain':meta['p95_abs_error']['R']-meta['p95_abs_error']['RM'],'p99_gain':meta['p99_abs_error']['R']-meta['p99_abs_error']['RM']}).reset_index(drop=True)
rows=sample.groupby('cap').actual_rows.sum().reset_index(name='unique_development_training_rows');rows['rows_across_three_folds_per_configuration']=2*rows.unique_development_training_rows
rows.to_csv(out/'training_volume_summary.csv',index=False)
timing=json.loads((out/'timing.json').read_text());reportonly=json.loads((out/'report_only_timing.json').read_text());case=json.loads((out/'case_inference_timing.json').read_text())
def tab(x):return x.to_markdown(index=False,floatfmt='.3f')
positives=days[(days.model=='R')&(days.gain>0)]
contents=['## 핵심 질문별 판정과 수치',
'1. **Residual 대 absolute:** 전체 run-balanced MAE 이점은 cap100/300/600에서 모두 유지됐다. 차이는 각각 240.009/177.004/125.738 W다. 그러나 cap100의 09-20, cap600의 09-16에서는 R이 A보다 나빴다. 따라서 H1은 날짜까지 고려하면 혼합이다.',tab(summary),
'2. **Persistence 초과 여부:** 전체 OOF에서는 R이 어떤 cap에서도 넘지 못했다. R MAE는 769.754→693.568→690.984 W로 감소했지만 Persistence 521.061 W보다 모두 높았다. 날짜별로 R이 Persistence보다 낮은 것은 09-20·cap300뿐(800.186 대 827.207 W, 3.267% 개선)이다. 09-20·cap600 R은 0.386 W 더 나빴다. RM은 09-20·cap600에서 1.472% 개선됐다. H2의 전체 평균 개선은 지지하되 날짜별·tail까지 일관된 개선은 아니다.',tab(positives[['cap','collection_day','mae','persistence_mae','improvement_pct']]),
'3. **구간 손익:** R의 정상 구간 전체 MAE 손해 기여는 −267.386→−206.831→−189.913 W로 줄었다. 하강 이득 기여는 +41.417→+39.896→+30.047 W로 줄었으며, 상승은 −22.723→−5.572→−10.056 W 손해다. 큰 변화는 겹치는 보조 구간이며 gain을 기여 합에 더하지 않는다. 아래 큰 변화 gain은 해당 구간의 run 평균이다. 정상 손해 감소만으로 전체 Persistence 초과를 이루지는 못했다.',tab(normal),tab(big),
'4. **Metadata 평균·tail:** R−RM 평균 MAE gain은 +2.732/−4.130/−12.928 W다. p95는 모든 cap에서 개선됐지만 p99는 모든 cap에서 악화됐다. Tail 전체가 개선됐다고 할 수 없다.',tab(effect),
'5. **날짜 일관성:** Metadata 평균 효과는 동일 방향이 아니다. Cap100에서 09-17만 개선, cap300에서 09-16/17 개선·09-20 악화, cap600에서는 09-16/17 악화·09-20 개선이다. Hardware/workload 효과의 인과 해석은 날짜 교락으로 불가능하다.',tab(pd.read_csv(out/'paired_metadata_effect_by_day.csv')[['cap','collection_day','mae_effect','p95_abs_error_effect','p99_abs_error_effect']]),
'6. **표본 증가로 개선된 부분과 남은 문제:** Cap100→300에서 R 전체 MAE가 76.186 W 감소했고 300→600 감소는 2.584 W에 그쳤다. 정상 불필요 보정 손해는 줄었으나 09-16/17의 Persistence 열세, 09-20의 비단조 변화, p95/p99 증가와 metadata 날짜 의존성은 남았다. 같은 run 시점 확대가 새로운 독립 조건의 일반화를 입증하지 않는다.',
'7. **다음 단계:** 독립 session·날짜·hardware·workload를 확보하는 데이터 다양성 확대를 우선한다. 이후 독립 자료에서 metadata 효과를 검증한다. 추가 시점에 따른 이득 둔화와 날짜별 부호 변화가 근거다. Feature 개선·모델 확장 우월성은 이번 고정 실험에서 판단할 수 없으며 자동 추가 탐색은 수행하지 않았다.',
'## 실제 학습량·시간·완료 범위',tab(rows),
'전체 개발 run은 23개, 고정 평가 anchor는 13,338개다. Unique 학습 시점 수와 fold 반복 사용 행 수를 구분한다. A/R/RM 각각의 fold 학습행은 fit_registry.csv에 있다. 신규 본 fit 27회·pilot 재사용 0회·별도 smoke 1회·report-only 신규 fit 0회다. cap300의 새 예측은 기존 pilot A/R/RM 개발 예측과 1e−8 W 이내에서 동일해 비교 연속성을 확인했다.',
f"본 pipeline wall-clock {timing['wall_clock_seconds']:.3f}초; 표본/자료 전처리 {timing['preprocessing_seconds']:.3f}초; 27회 학습 합 {timing['fit_seconds']:.3f}초; 평가 prediction 합 {timing['prediction_seconds']:.3f}초; 집계·표·그래프·보고서 단계 {timing['report_seconds']:.3f}초(설명용 inference 포함). Report-only wall-clock {reportonly['wall_clock_seconds']:.3f}초, 신규 fit 0회; 설명용 case inference {case['prediction_seconds']:.3f}초. Hash 감사·기존 예측 진단 등은 wall-clock에 포함되며 위 학습 합으로 대체하지 않는다. HTML 실제 렌더링과 별도 원본 대조 시간은 pipeline wall-clock 밖에 기록한다.",
'필수 산출물·27 cells·중첩·split·가중치·원본 예측 대조·구간 분해·PNG/SVG/CSV가 완료됐다. 독립 session 여부는 자료 부족이며 일반화·인과 검증은 이번 범위 밖이다. 제어 실행과 09-22 신규 모델 평가는 수행하지 않았다.',
'## 기존 저장 예측의 추가 크기 진단',tab(pd.read_csv(out/'prior_magnitude_diagnostics_summary.csv'))]
marker='\n\n<!-- verified supplement -->\n\n';base=(out/'report_ko.md').read_text().split(marker)[0];md=base+marker+'\n\n'.join(contents);(out/'report_ko.md').write_text(md)
html=markdown.markdown(md,extensions=['tables','fenced_code']);(out/'report_ko.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>AIDC residual metadata follow-up</title><style>body{font-family: NanumGothic,sans-serif;margin:32px;line-height:1.6}img{max-width:100%;height:auto}table{display:block;overflow-x:auto;border-collapse:collapse;max-width:100%}td,th{padding:6px;border:1px solid #ddd}h1,h2{line-height:1.3}</style>'+html+'</html>')
atomic_json(out/'finalization_status.json',{'status':'completed','fits':0,'elapsed_seconds':time.perf_counter()-start,'verification_inference_seconds':inference_seconds,'code_hash':digest(__file__)})
print(json.dumps({'status':'verified_and_finalized','fits':0,'elapsed_seconds':time.perf_counter()-start}))
