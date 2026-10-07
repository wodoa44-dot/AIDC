"""Regenerate figures and Korean report exclusively from saved aggregates."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
LABEL={'whole':'전체','normal':'정상','rising':'상승','falling':'하강','large_change':'큰 변화'}

def render(out):
    start=time.perf_counter()
    summary=pd.read_csv(out/'metrics_summary.csv');days=pd.read_csv(out/'metrics_by_day.csv');regions=pd.read_csv(out/'region_metrics.csv')
    dc=pd.read_csv(out/'gain_decomposition.csv');effect=pd.read_csv(out/'metadata_effect.csv');per=pd.read_csv(out/'metrics_per_run.csv');prov=pd.read_csv(out/'provenance.csv');v=json.loads((out/'verification.json').read_text())
    figs=out/'figures';plots=out/'plotting_data';figs.mkdir(exist_ok=True);plots.mkdir(exist_ok=True)
    names=[]
    def save(name,fig,d):
        d.to_csv(plots/(name+'.csv'),index=False);fig.tight_layout()
        fig.savefig(figs/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(figs/(name+'.svg'),bbox_inches='tight');plt.close(fig);names.append(name)
    def lines(ax,d,metric='mae',reference=True):
        for model in ['R','RM']:
            z=d[d.model==model].sort_values('lambda');ax.plot(z['lambda'],z[metric],marker='o',label=model)
        if reference:
            p=d[(d.model=='R')&(d['lambda']==0)][metric].iloc[0];ax.axhline(p,color='gray',ls='--',label='Persistence')
            a=d[d.model=='A']
            if len(a):ax.axhline(a[metric].iloc[0],color='orange',ls=':',label='Absolute (참고)')
        ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('λ');ax.grid(alpha=.25);ax.legend(fontsize=8)
    for metric,title,name in [('mae','Run-balanced MAE (W)','lambda_mae'),('p95_abs_error','Run별 p95 동일 가중 평균 (W)','lambda_p95'),('p99_abs_error','Run별 p99 동일 가중 평균 (W)','lambda_p99')]:
        fig,ax=plt.subplots(figsize=(7,4));lines(ax,summary,metric);ax.set_ylabel(title);save(name,fig,summary)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,(day,g) in zip(axes,days.groupby('collection_day')):
        lines(ax,g);ax.set_title(day);ax.set_ylabel('MAE (W)')
    save('day_mae',fig,days)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,region in zip(axes.flat,['normal','rising','falling','large_change']):
        z=regions[regions.region==region];lines(ax,z,'gain',False);ax.axhline(0,color='gray',ls='--');ax.set_title(LABEL[region]);ax.set_ylabel('Persistence 대비 MAE gain (W)')
    save('region_gain',fig,regions)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,model in zip(axes,['R','RM']):
        for region in ['normal','rising','falling']:
            z=dc[(dc.model==model)&(dc.region==region)].sort_values('lambda');ax.plot(z['lambda'],z.gain_contribution,marker='o',label=LABEL[region])
        z=summary[summary.model==model].sort_values('lambda');ax.plot(z['lambda'],z.gain,color='black',ls='--',label='합계')
        ax.set_title(model);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('λ');ax.set_ylabel('전체 MAE gain 기여도 (W)');ax.axhline(0,color='gray');ax.legend();ax.grid(alpha=.25)
    save('gain_contributions',fig,dc)
    fig,axes=plt.subplots(2,2,figsize=(11,7));z=effect[effect.scope=='whole'].sort_values('lambda')
    for ax,m in zip(axes.flat,['mae','rmse','p95_abs_error','p99_abs_error']):
        ax.plot(z['lambda'],z[m+'_effect'],marker='o');ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('λ');ax.set_ylabel('R - RM (W)');ax.set_title(m);ax.axhline(0,color='gray',ls='--');ax.grid(alpha=.25)
    save('metadata_effect',fig,effect)
    fig,axes=plt.subplots(1,2,figsize=(11,8));z=per[per.model.isin(['R','RM'])]
    limit=max(abs(z.gain).max(),1)
    for ax,model in zip(axes,['R','RM']):
        table=z[z.model==model].pivot(index='run_id',columns='lambda',values='gain').sort_index()
        mesh=ax.imshow(table,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit,extent=(-.125,1.125,len(table)-.5,-.5))
        ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('λ');ax.set_yticks(range(len(table)));ax.set_yticklabels([r[-8:] for r in table.index]);ax.set_title(model+' run별 gain');fig.colorbar(mesh,ax=ax,label='Persistence 대비 gain (W)')
    save('run_gain_heatmap',fig,z)
    visualization_seconds=time.perf_counter()-start
    report_start=time.perf_counter()
    def table(d,cols=None):
        if cols:d=d[cols]
        return d.to_markdown(index=False,floatfmt='.3f',missingval='N/A').replace('nan','N/A')
    def val(model,lam,region=None,metric='mae'):
        d=summary if region is None else regions
        z=d[(d.model==model)&(d['lambda']==lam)]
        if region is not None:z=z[z.region==region]
        return float(z.iloc[0][metric])
    q=[]
    for model in ['R','RM']:
        g=summary[summary.model==model].sort_values('lambda');best=g.loc[g.mae.idxmin()]
        q.append(f"{model}: 다섯 계수 중 관측 최저 MAE는 λ={best['lambda']:g}, {best.mae:.3f} W이다. Persistence 대비 gain은 {best.gain:.3f} W이다.")
    obs=[]
    inter=summary[summary.model.isin(['R','RM'])&summary['lambda'].isin([.25,.5,.75])]
    obs.append(('중간 계수가 Persistence보다 좋은가?',f"중간 계수의 최고 gain은 {inter.gain.max():.3f} W. "+('개선이 관찰된다.' if (inter.gain>0).any() else '모든 중간 계수가 전체 MAE에서 Persistence보다 나쁘다.'),'개발 OOF 재사용 진단이며 독립 일반화 평가가 아니다.'))
    dd=days[days.model.isin(['R','RM'])&days['lambda'].isin([.25,.5,.75])]
    obs.append(('날짜별 방향이 같은가?',f"중간 계수의 날짜별 양수 gain cell은 {(dd.gain>0).sum()}/{len(dd)}개. 날짜별 수치는 아래 표에 제시한다.",'9월 16·17일은 모든 중간 계수에서 손해, 20일은 R/RM 모두 이득이다. 날짜별 방향은 일관적이지 않다. 최저 계수 차이는 gate 학습 가능성의 근거가 아니다.'))
    normal=[];large=[]
    for model in ['R','RM']:
        normal.append(f"{model} 정상 gain λ=1 {val(model,1,'normal','gain'):.3f} → λ=0.25 {val(model,.25,'normal','gain'):.3f} W")
        large.append(f"{model} 큰 변화 gain λ=1 {val(model,1,'large_change','gain'):.3f} → λ=0.25 {val(model,.25,'large_change','gain'):.3f} W")
    obs.append(('정상 손해 감소와 큰 변화 이득 보존?', '; '.join(normal+large),'endpoint 변화로 나눈 사후 구간이다. 미래 경로·전이 시작 예측 성능으로 해석하지 않는다.'))
    tails=[]
    for model in ['R','RM']:
        tails.append(f"{model} p95: λ=1 {val(model,1,metric='p95_abs_error'):.3f}, 0.25 {val(model,.25,metric='p95_abs_error'):.3f}, 0 {val(model,0,metric='p95_abs_error'):.3f}; p99: {val(model,1,metric='p99_abs_error'):.3f}, {val(model,.25,metric='p99_abs_error'):.3f}, {val(model,0,metric='p99_abs_error'):.3f} W")
    obs.append(('p95·p99는 어떻게 달라지는가?', '; '.join(tails),'p95·p99는 λ=1부터 축소할 때 처음에 낮아졌다가 더 축소하면 다시 높아진다. Persistence의 꼬리 오차보다 낮은 중간 계수가 있지만 전체 MAE 개선과 일치하지 않는다. 분위수에는 convexity 조건을 적용하지 않는다.'))
    ee=effect[effect.scope=='whole']
    obs.append(('동일 계수 metadata 효과?',f"0 외 λ에서 전체 MAE 효과 범위 {ee[ee['lambda']>0].mae_effect.min():.3f}~{ee[ee['lambda']>0].mae_effect.max():.3f} W. λ=0 효과는 정확히 0.",'λ=0.25·0.5에서는 작은 MAE 개선, 0.75·1에서는 악화다. 날짜·구간별 부호도 달라 일관된 metadata 이득은 없다. Signed bias 차이의 양수는 개선이 아니다.'))
    conclusion=('일괄 축소 가능성 관찰' if (inter.gain>0).any() else '이번 범위에서 Persistence 초과 근거 없음')
    obs.append(('일괄 축소 또는 조건별 검토?',conclusion+'; 정상·변화 및 날짜별 손익 충돌은 기여도와 구간 표로 확인한다.','과대 보정을 유일 원인으로 단정하지 않는다. Persistence가 최저여도 전력 예측 불가능성을 뜻하지 않는다.'))
    md='# AIDC residual 보정 계수 진단\n\n'
    md+=f"분석 완료: **{v['actual_runs']}개 run · {v['actual_anchors']:,} anchors**, 신규 fit **0회**, 추가 inference **0회**. 2025-09-16·17·20의 기존 leave-day-out 개발 OOF만 사용했다. 2025-09-22 자료는 평가하거나 행으로 로드하지 않았다. **{conclusion}**.\n\n"
    md+=' '.join(q)+'\n\n'
    md+='## 1. 입력 provenance와 고정 범위\n\n'
    md+='모든 원본은 보존했다. PROJECT_ROOT는 `/home/work/AIDC`이며 Git 저장소가 아니고 적용 AGENTS.md는 없다. 실제 경로와 SHA-256은 `resolved_paths.json`, `input_manifest.json`에 기록했다. fit의 training/evaluation anchor hash, 예측·모델·weight 파일 hash를 확인했으며 모델을 로드하거나 재추론하지 않았다.\n\n'
    md+=''+table(prov,['model','fold','train_days','evaluation_day','train_runs','train_anchors','n_runs','n_anchors','q50_w','q90_w','unit'])+'\n\n'
    md+='training cap은 run당 최대 300, history는 현재 완료 bin을 포함한 25 bins(5초), 입력 bin은 200 ms, horizon은 1초다. Target은 발행 t에 대해 `(t+0.8,t+1.0]`의 8-GPU 합산 전력 bin 평균이다. 원본 preprocessing은 센서 표본의 산술 평균을 사용한다. timestamp는 run 상대 초이며 key는 `(run_id, segment, bin)`이다. R/RM/A는 key로 manifest와 결합하여 현재값·target·fold·시각을 대조했다.\n\n'
    md+='원본 pipeline은 R/RM에서 `current + raw residual`을 저장하며 clipping·smoothing이 없다. 별도 raw residual 벡터가 없어 `saved_prediction-current`로 복원했다. 새 후처리 없이 `current + λ*(saved_prediction-current)`만 계산했다. 사전 고정 λ는 0, 0.25, 0.5, 0.75, 1이다. Absolute는 같은 표본의 참고선이고 계수 실험 대상이 아니다.\n\n'
    md+='## 2. 전체 성능\n\n'+table(summary,['model','lambda','mae','rmse','signed_bias','p95_abs_error','p99_abs_error','gain','improvement_pct','gain_vs_lambda1','improved_runs','tied_runs','worsened_runs','n_runs','n_anchors','p99_valid_runs','unit'])+'\n\n'
    md+='오차·bias·gain의 단위는 W이고 improvement_pct는 %. Run별 지표를 먼저 계산해 각 run을 동일 가중 평균했다. RMSE는 run별 RMSE의 평균이며 전체 pooled RMSE가 아니다. p99는 해당 구간 n<100이면 N/A, 빈 구간의 지표도 N/A로 두었다. n_runs는 전체 run 수, valid_region_runs는 비어 있지 않은 run 수이며 p99_valid_runs는 p99를 계산할 수 있는 run 수다. 동률 판정 허용오차는 1e-9 W다.\n\n'
    md+='## 3. 날짜별 성능\n\n'+table(days,['collection_day','model','lambda','mae','gain','improvement_pct','p95_abs_error','p99_abs_error','n_runs','n_anchors','p99_valid_runs','unit'])+'\n\n'
    md+='## 4. 구간별 성능\n\n'
    md+='q50·q90은 각 fold의 기존 cap300 training anchors에서 NumPy linear quantile로 복원하여 저장 threshold와 일치함을 확인했다. 정상은 |actual−current|≤q50, 상승은 actual−current>q50, 하강은 actual−current<−q50, 큰 변화는 |actual−current|>q90이다. 모든 계수·모델에 같은 membership을 적용했다. 큰 변화는 상승·하강과 중첩되는 보조 구간이다.\n\n'
    md+=table(regions[regions.region!='whole'],['region','model','lambda','mae','rmse','signed_bias','p95_abs_error','p99_abs_error','gain','gain_vs_lambda1','n_runs','valid_region_runs','n_anchors','p99_valid_runs','unit'])+'\n\n'
    md+='## 5. 전체 gain 분해\n\n'
    md+='각 run의 정상·상승·하강 anchor gain 합계를 **해당 run 전체 anchor 수**로 나누고 run 간 평균했다. 구간별 성능 표는 각 구간 안에서의 평균이므로 기여도와 다른 집계다. 빈 구간 기여도는 빈 합계로서 0이며, 빈 구간 성능을 0으로 대체하지 않았다. 세 기여도의 합은 전체 MAE gain이다. 큰 변화는 다시 더하지 않는다.\n\n'
    md+=table(dc,['model','lambda','region','gain_contribution','n_runs','n_anchors','whole_n_anchors','unit'])+'\n\n'
    md+='## 6. 동일 계수 metadata 효과\n\n'
    md+='`effect = metric_R − metric_RM`. MAE·RMSE·p95·p99에서 양수는 metadata 개선이다. signed_bias_effect는 단순 차이이며 양수를 개선으로 부르지 않는다. 날짜×구간 상세는 `metadata_effect.csv`에 있다.\n\n'
    md+=table(effect[effect.scope.isin(['whole','day','region'])],['scope','collection_day','region','lambda','mae_effect','rmse_effect','signed_bias_effect','p95_abs_error_effect','p99_abs_error_effect','n_runs','n_anchors','valid_region_runs','p99_valid_runs','unit'])+'\n\n'
    md+='## 7. 수치 검증\n\n'
    verify=pd.DataFrame([dict(check=k,result=str(x),unit='W' if k.endswith('_w') else '—',n_runs=v['actual_runs'],n_anchors=v['actual_anchors']) for k,x in v.items()])
    md+=table(verify)+'\n\n'
    md+='반올림 보고서 숫자는 입력이 아니다. 원본 수치 CSV와 λ=1의 run별 지표 및 전체 지표를 비교했다. 같은 표본·고정 가중치에서 전체·날짜·구간·날짜×구간·run×구간 MAE의 2차 차분을 확인했다. 저장 표본을 임의로 23/13,338에 맞추지 않았다.\n\n'
    md+='## 8. 질문별 관찰·해석·한계\n\n'+table(pd.DataFrame(obs,columns=['질문','관찰·해석','한계']))+'\n\n'
    md+='개발 자료를 반복 사용한 탐색 결과이며 최저 MAE 계수는 관측 기술에 한정한다. 최종 모델·검증된 최적 계수로 선택하지 않았다. 사후 미래 label로 구간을 나눈 것이며 oracle 선택, gate 구현·학습, 추가 계수 탐색, 재학습, 보류 날짜 평가는 수행하지 않았다. 필요시 조건별 보정은 향후 독립 설계·검증 과제이지 이번 결과로 실행 가능한 gate 성능을 주장할 수 없다.\n\n'
    md+='## 9. 그래프와 재현\n\n'
    for name in names:md+=f'![{name}](figures/{name}.png)\n\n'
    md+='각 그래프의 PNG·SVG 및 plotting CSV를 저장했다. λ 축은 실제 수치 간격이며 불연속 anchors를 시계열 파형으로 연결하지 않았다. Heatmap y축은 run_id 마지막 8자이고 전체 key는 plotting CSV에 있다.\n\n'
    md+='`pipeline.log`, `process_identity.json`, `pipeline.pid`, `progress.json`, `completion_status.json`, `exit_code.txt`로 nohup 실행을 추적한다. 전체 및 단계별 시간은 `timing.json`, 그림·보고서 전용 재생성 시간은 `report_only_timing.json`에 기록한다. `README.md`의 명령으로 재현하며 resume는 입력·코드·config hash가 일치할 때만 허용한다.\n'
    (out/'report_ko.md').write_text(md)
    html=markdown.markdown(md,extensions=['tables','fenced_code'])
    (out/'report_ko.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>AIDC 보정 계수 진단</title><style>body{font-family:"NanumGothic",sans-serif;margin:28px;line-height:1.6}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{border:1px solid #ccc;padding:5px;white-space:nowrap}img{max-width:100%}h2{margin-top:32px}</style><body>'+html+'</body></html>')
    assert len(list(figs.glob('*.png')))==8 and len(list(figs.glob('*.svg')))==8
    for name in names:assert (plots/(name+'.csv')).exists()
    write_result=dict(figures_png_svg_pairs=8,plotting_csv_count=8,korean_font='NanumGothic',html_image_count=html.count('<img'),report_generated_from_saved_aggregates=True)
    (out/'render_verification.json').write_text(json.dumps(write_result,ensure_ascii=False,indent=2))
    return dict(visualization_seconds=visualization_seconds,report_generation_seconds=time.perf_counter()-report_start)
