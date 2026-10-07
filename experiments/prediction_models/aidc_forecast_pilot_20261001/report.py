"""Tables and explanatory plots from frozen prediction-only pilot artifacts."""
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
from data import columns,encode,KEY


def render(cfg,out):
    figures=out/'figures';figures.mkdir(exist_ok=True);plotcsv=out/'plotting_csv';plotcsv.mkdir(exist_ok=True)
    summary=pd.read_csv(out/'aggregate_metrics.csv');runs=pd.read_csv(out/'per_run_metrics.csv');selected=json.loads((out/'selected_configuration.json').read_text());fit_summary=json.loads((out/'fit_summary.json').read_text())
    dev=summary[(summary.scope=='development')&(summary.region=='whole')&(summary.postprocess=='raw')].copy()
    hold=summary[(summary.scope=='holdout')&(summary.region=='whole')&(summary.postprocess=='raw')].copy()
    region=summary[(summary.scope=='holdout')&(summary.postprocess=='raw')].copy()
    dev.to_csv(out/'development_table.csv',index=False);hold.to_csv(out/'holdout_table.csv',index=False)
    # Numeric history axes, no artificial model ordering in line charts.
    caption='GPU sum power endpoint | pilot, reused exploratory evaluation | no CI or significance tests\nEqual weight per run; assumed calendar groups; overlapping histories are not independent samples'
    def save(fig,name,data):
        fig.text(.5,.01,caption,ha='center',fontsize=7);fig.tight_layout(rect=(0,.08,1,1))
        fig.savefig(figures/f'{name}.png',dpi=300,bbox_inches='tight');fig.savefig(figures/f'{name}.svg',bbox_inches='tight');plt.close(fig);data.to_csv(plotcsv/f'{name}.csv',index=False)
    history=dev[dev.model.str.match(r'^(absolute|residual)_h\d+$')].copy();history['history_s']=history.model.str.extract(r'_h(\d+)').astype(int);history['target']=history.model.str.split('_').str[0]
    for metric,name in [('mae','history_mae'),('p95_abs_error','history_p95')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for target,g in history.groupby('target'):
            g=g.sort_values('history_s');ax.plot(g.history_s,g[metric],'o-',label=target)
        ax.axhline(dev[dev.model=='persistence'][metric].iloc[0],color='black',linestyle='--',label='persistence')
        ax.set(xlabel='History seconds',ylabel=metric+' (W)',title='Development OOF: 3 date validation folds');ax.set_xticks([1,5,10]);ax.legend();save(fig,name,history)
    fig,ax=plt.subplots(figsize=(10,5));ax.barh(hold.model,hold.mae_improvement_pct);ax.axvline(0,color='gray');ax.set(xlabel='MAE improvement over Persistence (%)',title='Held-out 2025-09-22; equal-run aggregate');save(fig,'holdout_improvement',hold)
    h=runs[(runs.scope=='holdout')&(runs.region=='whole')&(runs.postprocess=='raw')];p=h.pivot(index='run_id',columns='model',values='mae')
    pdiff=p.subtract(p.persistence,axis=0).drop(columns='persistence').reset_index();fig,ax=plt.subplots(figsize=(11,5))
    for model in pdiff.columns[1:]:ax.scatter(pdiff.run_id.str[-8:],pdiff[model],label=model)
    ax.axhline(0,color='gray');ax.set(xlabel='Run ID suffix (unordered)',ylabel='Model MAE - Persistence MAE (W)');ax.tick_params(axis='x',rotation=30);ax.legend(fontsize=8);save(fig,'run_mae_difference',pdiff)
    inventory=pd.read_csv(out/'run_inventory.csv');full=pd.read_parquet(out/'continuous_features.parquet');train=pd.read_parquet(out/'train_anchors.parquet')
    final_models={}
    for folder in (out/'fits').glob('holdout_*'):
        rec=json.loads((folder/'complete.json').read_text());final_models[rec['identity']['target']+'_h'+str(rec['identity']['history'])+('_metadata' if rec['identity']['metadata'] else '')]=(folder,rec)
    ranking=pd.DataFrame({'delta':p[selected['model']]-p.persistence}).reset_index().sort_values(['delta','run_id'])
    cases=[]
    for label,idx in [('improved',0),('middle',len(ranking)//2),('worsened',len(ranking)-1)]:
        run=ranking.iloc[idx];g=full[(full.run_id==run.run_id)&full.valid_anchor].sort_values('issuance_s')
        breaks=np.r_[True,(g.segment.to_numpy()[1:]!=g.segment.to_numpy()[:-1]) | ~np.isclose(np.diff(g.issuance_s),.2,atol=1e-7,rtol=0)]
        blocks=[a for _,a in g.groupby(np.cumsum(breaks))];x=sorted(blocks,key=lambda a:(-len(a),a.issuance_s.iloc[0]))[0].head(100).copy()
        assert np.allclose(np.diff(x.issuance_s),.2,atol=1e-7)
        z=x[KEY+['issuance_s','actual_w','current_w']].copy();z['target_timestamp']=z.issuance_s+1
        z['persistence']=z.current_w
        for model,(folder,rec) in final_models.items():
            _,xv,_,_=encode(train,x,rec['feature_columns'],rec['categorical_encoding'])
            output=lgb.Booster(model_file=str(folder/'q50.txt')).predict(xv,num_threads=cfg['threads'])
            if rec['identity']['target']=='residual':output+=x.current_w.to_numpy()
            z[model]=output
        fig,ax=plt.subplots(figsize=(11,5));t=z.target_timestamp-z.target_timestamp.iloc[0]
        ax.plot(t,z.actual_w,label='actual target',color='black',linewidth=2)
        for model in ['persistence',*final_models]:ax.plot(t,z[model],label=model,alpha=.85)
        ax.set(xlabel='Target end time relative to first plotted target (s)',ylabel='GPU sum power (W)',title=f'{label} rank: {run.run_id} | whole-run MAE delta {run.delta:.1f} W');ax.legend(fontsize=7);save(fig,'case_'+label,z)
        mse=lambda a:float(np.mean(abs(z[a]-z.actual_w)))
        cases.append({'rank_label':label,'run_id':run.run_id,'whole_run_selected_minus_persistence_mae_w':run.delta,'plotted_segment':int(x.segment.iloc[0]),'first_issuance_s':float(x.issuance_s.iloc[0]),'last_issuance_s':float(x.issuance_s.iloc[-1]),'continuous_plot_n':len(x),'plotted_selected_mae_w':mse(selected['model']),'plotted_persistence_mae_w':mse('persistence'),'rule':'min/median/max holdout whole-run MAE delta, then first 100 valid anchors of longest continuous block; rank labels may have same sign; no tuning'})
    pd.DataFrame(cases).to_csv(out/'case_selection.csv',index=False)
    # Explicitly record even empty endpoint regions, all models share fold thresholds.
    counts=[];recorded=set()
    for folder in sorted((out/'fits').iterdir()):
        rec=json.loads((folder/'complete.json').read_text());identity=rec['identity'];key=(identity['scope'],str(identity['fold']))
        if key in recorded:continue
        recorded.add(key);d=pd.read_parquet(folder/'predictions.parquet');g=d[(d.region=='whole')&(d.postprocess=='raw')];delta=g.actual_w-g.current_w
        q50=rec['threshold_q50_w'];q90=rec['threshold_q90_w']
        for name,mask in [('normal',abs(delta)<=q50),('rising',delta>q50),('falling',delta < -q50),('large_endpoint_change',abs(delta)>q90)]:counts.append({'scope':identity['scope'],'fold':identity['fold'],'region':name,'n':int(mask.sum()),'q50_w':q50,'q90_w':q90,'empty':bool(not mask.any())})
    pd.DataFrame(counts).to_csv(out/'endpoint_region_counts.csv',index=False)
    for _,r in hold.iterrows():
        per=h[h.model==r.model]
        assert np.isclose(per.mae.mean(),r.mae)
        assert int(per.n.sum())==r.n_samples
    tablecols=['model','mae','rmse','signed_bias','p95_abs_error','p99_abs_error','mae_improvement_pct','n_runs','n_samples','p99_n_runs','improved_runs','worsened_runs']
    base=hold[hold.model=='persistence'].iloc[0];chosen=hold[hold.model==selected['model']].iloc[0];metarow=hold[hold.model==selected['model']+'_metadata'].iloc[0]
    histbase=dev[dev.model=='absolute_h1'].iloc[0];devchosen=dev[dev.model==selected['model']].iloc[0]
    absbest=history[history.target=='absolute'].sort_values(['mae','p95_abs_error','history_s']).iloc[0];resbest=history[history.target=='residual'].sort_values(['mae','p95_abs_error','history_s']).iloc[0]
    conclusion='확대 검증할 가치 있음' if chosen.mae<base.mae and metarow.mae<base.mae else ('조건별로 달라 추가 확인 필요' if devchosen.mae < histbase.mae or chosen.mae<base.mae or metarow.mae<chosen.mae else '이 pilot에서는 개선 근거 없음')
    audit=[]
    for scope in ['development','holdout']:
        file=out/f'legacy_audit_{scope}/summary.csv'
        if file.exists():audit.append(pd.read_csv(file).assign(scope=scope))
    audit_table=pd.concat(audit,ignore_index=True) if audit else pd.DataFrame({'status':['unavailable; see availability.csv']})
    text=f'''# AIDC 예측모델 소규모 pilot

{conclusion}. 개발 OOF에서 선택한 구성은 **{selected['model']}**이다. 이 보고서는 GPU 합산 전력의 1초 endpoint q50 예측만 평가한다. 제어·MPC·저장장치·계통 시뮬레이션은 실행하지 않았다. 기존 평가셋 재사용과 보류 날짜 하나의 탐색적 pilot이며, 독립 일반화 검증이 아니다.

## 표본과 split

개발은 2025-09-16·17·20의 leave-day-out 3개 fold, 최종 fit은 이 3개 날짜를 학습하고 2025-09-22를 보류 평가한다. 수집 날짜는 session 독립성이 확인된 group이 아니며 hardware/workload와 교락되어 있다. Run ID·날짜·파일명은 feature가 아니다.

공통 anchor는 50개 완료 bin의 history와 다음 5개 bin 모두 유효하고 연속인 위치다. History=1/5/10초는 각각 현재를 포함한 5/25/50개 bin을 사용한다. 기존 feature에서 lag가 history bin 수와 같은 경계 항목은 추가 과거 bin을 읽으므로 새 pilot에서는 제외했다. 기존 모델 정렬 감사에서는 원래 lag·encoding을 그대로 복원했다. 결측 보간은 없다.

각 segment 첫 유효 anchor에서 최소 1초 간격으로 후보를 만든다. Training은 run당 300행 이내, evaluation은 600행 이내이며 시간 인덱스 10개 층으로 선택했다. 30개 미만 후보 run은 제외했다. 10초 history가 겹치므로 표본은 독립이 아니다. 학습 가중치 N/(R×n_r)를 사용해 run 총 가중치를 같게 했다. Feature는 원래 연속 bin에서 계산한 후 표본을 골랐다.

'''
    text+=inventory[['run_id','collection_day','original_bins','valid_anchors','spaced_candidates','training_rows','evaluation_rows','included','exclusion_reason']].to_markdown(index=False)
    text+='\n\n세부 split은 split_manifest.json, 공통 key와 추출 규칙은 train_anchors.parquet 및 evaluation_anchors.parquet에 있다. 전체 32개 후보 중 포함·제외 기준은 성능과 무관하다. 보류 run은 training_rows=0이다.\n\n## 기존 q50 정렬 감사\n\n'
    text+=audit_table.to_markdown(index=False,floatfmt='.3f')
    text+='\n\n각 날짜의 기존 outer-test 모델을 사용하여 평가 run 학습 중복이 없는지, 학습 encoding hash와 모델 checksum 및 sample key·target·저장 sorted median 일치를 확인했다. 기존 30초 history eligibility 때문에 새 pilot anchor 일부만 겹치며 audit availability.csv에 범위를 저장했다. 기존 모델의 학습량·split은 새 pilot과 다르고 개발 audit 모델에는 pilot 보류 날짜가 학습에 포함될 수 있다. 이 표를 새 pilot의 순수 독립 성능 비교로 쓰지 않는다. 원래 q50과 정렬 후 median의 차이는 anchor_outputs.parquet, run별 개선·악화는 per_run.csv에서 확인한다. 15 quantile 재학습은 없다.\n\n## Target와 history 개발 결과\n\n'
    text+=dev[tablecols].to_markdown(index=False,floatfmt='.3f')
    text+=f'\n\n모든 fold OOF의 run별 지표를 먼저 계산해 run을 동일 가중으로 평균했다. 서로 다른 크기의 fold 평균을 동일 가중하지 않았다. 선정 MAE는 {selected["development_mae"]:.3f} W이다. 개발 best residual MAE={resbest.mae:.3f} W, best absolute MAE={absbest.mae:.3f} W로 비교한다. 구성 선택 이후 metadata 3회만 추가 학습했으며 같은 개발 자료를 다시 쓴 metadata 비교는 탐색적이다.\n\n![history MAE](figures/history_mae.png)\n\n실제 1·5·10초 간격에서 absolute/residual을 비교한다. Persistence 수평선은 공통 anchor의 기준이다.\n\n![history p95](figures/history_p95.png)\n\n평균 오차와 꼬리 오차의 방향이 같은지 별도로 확인한다.\n\n## 보류 날짜 최종 평가\n\n'
    text+=hold[tablecols].to_markdown(index=False,floatfmt='.3f')
    text+=f'\n\n선택 구성 MAE={chosen.mae:.3f} W, metadata 추가 MAE={metarow.mae:.3f} W, Persistence MAE={base.mae:.3f} W다. 양의 개선율은 Persistence보다 낮은 MAE다. 보류 결과로 구성·표본·trees를 다시 선택하지 않았다. p99는 run 표본 100개 미만이면 N/A이며 p99_n_runs가 집계 기여 run 수다. Primary는 후처리 없는 예측이다. 동일 0 하한 clipping 진단은 aggregate_metrics.csv와 clipping_audit.csv에서 확인한다. 상한 clipping은 없다.\n\n![보류 개선율](figures/holdout_improvement.png)\n\n범주형 모델명을 선으로 연결하지 않고 막대로 비교했다.\n\n![run별 MAE 차이](figures/run_mae_difference.png)\n\n음수는 Persistence보다 낮은 오차이며 특정 run 집중 여부를 읽을 수 있다.\n\n## Endpoint 변화 구간\n\n'
    text+=region[['model','region','mae','p95_abs_error','n_runs','n_samples','p99_n_runs']].to_markdown(index=False,floatfmt='.3f')
    text+='\n\n각 training의 |future−current| q50/q90을 고정해 정상·상승·하강·큰 변화를 평가했다. q50 경계는 정상에 포함하고 큰 변화는 다른 구간과 중첩된다. 이는 endpoint 변화이며 1초 경로 최대 ramp가 아니다. endpoint_region_counts.csv는 빈 구간도 0으로 기록한다. Threshold는 region label에만 쓰고 학습 feature나 표본 선택에 쓰지 않았다. sampling_region_audit.csv는 training-only threshold로 추출 전후 구성비를 보존하며 이를 보고 재추출하지 않았다.\n\n## Metadata와 clipping 감사\n\nMetadata는 기존 raw 경로 토큰에서 추출한 hardware·workload·model_size_token·image_size·batch_size·sequence_length·zero_stage를 사용한다. 실제 컬럼은 모두 존재하지만 값이 없거나 상수인 열이 있으며 추정해 채우지 않았다. 범주 mapping은 fit별 training에서만 학습하고 unseen/결측 범주는 −1, 숫자 결측은 LightGBM의 NaN으로 전달한다. encoding_diagnostics.csv는 fit·열별 unseen 비율·결측률·상수 여부를 기록한다. feature_schema.json과 각 fit complete.json에 실제 목록·수·encoding이 있다.\n\n'
    enc=pd.read_csv(out/'encoding_diagnostics.csv');text+=enc[(enc.scope=='holdout')&enc.column.isin(['hardware','workload']+columns(selected['history'],True)[-7:])].to_markdown(index=False,floatfmt='.3f')
    text+='\n\n## 연속 구간 사례\n\n개선·중간·악화 이름은 선택 구성의 전체 holdout run MAE−Persistence MAE 순위다. 부호가 같을 수 있으며 실제 성공·실패를 이름으로 가정하지 않는다. 각 run의 가장 긴 유효 연속 block 첫 100개 anchor에서 별도 inference했다. 불연속 평가 anchor를 파형으로 연결하지 않았으며 예측은 target 끝 시각에 배치했다. 표시 구간과 run 전체 지표를 아래에 따로 기록했다. 사례는 사후 설명용이며 tuning에 사용하지 않았다.\n\n'+pd.DataFrame(cases).to_markdown(index=False,floatfmt='.3f')
    for label in ['improved','middle','worsened']:text+=f'\n\n![{label} 사례](figures/case_{label}.png)\n\n{label} 순위의 연속 표시 구간이다. 이 짧은 구간이 전체 run을 대표한다고 가정하지 않는다.'
    text+=f'''\n\n## 최종 판단과 실행량

- 기존 q50 정렬 영향: 위 감사 표와 run별 raw/sorted 차이로 판정한다. 정렬과 새 target/history 효과는 서로 다른 비교다.
- Residual 가능성: 개발 best residual {resbest.mae:.3f} W 대 best absolute {absbest.mae:.3f} W다. 선택 구성의 보류 성능과 함께 읽어야 한다.
- History 증가: 선택 history는 {selected['history']}초이며, absolute 1초 개발 MAE {histbase.mae:.3f} W 대 선택 구성 {devchosen.mae:.3f} W다. target 효과와 history 효과의 결합은 6개 개별 결과로 구분한다.
- Metadata 가능성: 보류 metadata−history-only MAE={metarow.mae-chosen.mae:.3f} W다. 날짜 하나와 hardware/workload 교락으로 일반화하지 않는다.
- Persistence 초과 여부: 선택 구성 개선율 {chosen.mae_improvement_pct:.3f}%, metadata 개선율 {metarow.mae_improvement_pct:.3f}%다.
- 평균과 큰 변화: 전체와 large_endpoint_change 행을 각각 비교한다. 빈 구간은 판단 불가다. 이번에는 q50만 새로 학습해 새 모델 CRPS·coverage를 계산하지 않았다.
- 다음 확대 검증 후보: 개발 선택 {selected['model']}과 같은 구성의 metadata 모델을 후보로 남긴다. 보류 Persistence 대비 실패했다면 즉시 대규모 확대 근거로 삼지 않는다. 이번 비교가 끝난 뒤 자동으로 추가 모델·대규모 탐색을 시작하지 않는다.

판정: **{conclusion}**. 소량 학습의 실패는 LightGBM 계열 전체의 한계나 예측 불가능성의 증명이 아니다. 독립 수집 session·새 날짜에서 분할을 사전 고정한 추가 검증이 필요하다.

새 pilot q50 fit {fit_summary['new_q50_fits']}회, 최종 training {fit_summary['final_training_rows']}행/{fit_summary['final_training_runs']}run, run당 최대 {fit_summary['maximum_training_rows_per_run']}행, fit 및 prediction 합계 {fit_summary['fit_prediction_seconds']:.3f}초다. 별도 smoke는 100행 이하·5 trees·1 fit으로 성능표에서 제외했다. 전 과정 시간·exit code는 completion_status.json과 pipeline.log에 저장한다. Code/config/input identity가 일치할 때만 checkpoint를 재사용한다. 통계 bootstrap·정식 유의성 검정은 수행하지 않았다.
'''
    text=re.sub(r'\bnan\b','N/A',text)
    (out/'report_ko.md').write_text(text)
    html='<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>AIDC 예측모델 pilot</title><style>body{font-family: sans-serif;max-width:1200px;margin:24px auto;padding:16px;line-height:1.6;overflow-wrap:anywhere}table{display:block;overflow:auto;font-size:12px;border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}img{max-width:100%;height:auto}</style></head><body>'+markdown.markdown(text,extensions=['tables'])+'</body></html>'
    (out/'report_ko.html').write_text(html)
    (out/'report_checks.json').write_text(json.dumps({'aggregate_from_run_metrics_verified':True,'history_axes':[1,5,10],'continuous_inference_cases':3,'all_png_svg_csv_names':[p.stem for p in figures.glob('*.png')]},indent=2))

    with open(out/'render.log','w') as render_log:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('render_html.py')),str(out)],stdout=render_log,stderr=subprocess.STDOUT,timeout=60)
    assert result.returncode==0,'Actual HTML render failed; see render.log'
    check=json.loads((out/'render_verification.json').read_text())
    assert check['all_images_loaded'] and not check['horizontal_overflow']
