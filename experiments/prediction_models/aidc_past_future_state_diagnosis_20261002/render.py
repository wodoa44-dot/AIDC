"""Deterministic figures and Korean reports from saved aggregations."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
CELLS=[p+'_'+f for p in ['low','high'] for f in ['small','medium','large']]
LABEL={p+'_'+f:ps+'→'+fs for p,ps in [('low','저변동'),('high','고변동')] for f,fs in [('small','작은 변화'),('medium','중간 변화'),('large','큰 변화')]}

def render(out):
    out=Path(out);figdir=out/'figures';plotdir=out/'plotting_data'
    figdir.mkdir(exist_ok=True);plotdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':10})
    s=pd.read_csv(out/'metrics_summary_state.csv');day=pd.read_csv(out/'metrics_by_day_state.csv')
    counts=pd.read_csv(out/'state_counts.csv');dc=pd.read_csv(out/'gain_contributions.csv');eff=pd.read_csv(out/'metadata_effect.csv')
    base=pd.read_parquet(out/'anchor_states.parquet');primary=s[(s.history_seconds==5)&s.cell.isin(CELLS)]
    figs=[];checks=[]
    def save(name,fig,data):
        fig.tight_layout(pad=1.5)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();fw,fh=fig.get_size_inches()*fig.dpi
        outside=[];hidden_ticks=set()
        # Matplotlib retains tick artists beyond the viewing limits, which are
        # not drawn. Check only the tick labels that can actually be rendered.
        for ax in fig.axes:
            for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                lo,hi=sorted(limits)
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not lo<=tick.get_loc()<=hi:
                        hidden_ticks.update([id(tick.label1),id(tick.label2)])
        for text in fig.findobj(matplotlib.text.Text):
            if text.get_visible() and text.get_text() and id(text) not in hidden_ticks:
                b=text.get_window_extent(renderer)
                if b.x0< -2 or b.y0< -2 or b.x1>fw+2 or b.y1>fh+2:outside.append(text.get_text())
        assert not outside,(name,outside)
        for ext in ['png','svg']:fig.savefig(figdir/f'{name}.{ext}',dpi=160,bbox_inches='tight')
        data.to_csv(plotdir/f'{name}.csv',index=False)
        checks.append(dict(name=name,text_within_canvas=True,png_bytes=(figdir/f'{name}.png').stat().st_size,svg_bytes=(figdir/f'{name}.svg').stat().st_size))
        plt.close(fig);figs.append(name)
    def heat(ax,data,value,title,center=False):
        d=data.set_index('cell').reindex(CELLS);a=d[value].to_numpy().reshape(2,3)
        lim=max(1,np.nanmax(abs(a))) if np.isfinite(a).any() else 1
        im=ax.imshow(a,cmap='RdBu' if center else 'Blues',vmin=-lim if center else 0,vmax=lim)
        for i,r in enumerate(d.itertuples()):
            v=getattr(r,value);nr=getattr(r,'valid_runs',getattr(r,'n_runs',0));na=getattr(r,'n_anchors',0)
            ax.text(i%3,i//3,('N/A' if not np.isfinite(v) else f'{v:.1f}')+f'\n{int(nr)} runs / {int(na)} anchors',ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.75,edgecolor='none'))
        ax.set_xticks(range(3),['작은 변화','중간 변화','큰 변화']);ax.set_yticks(range(2),['과거 저변동','과거 고변동']);ax.set_title(title)
        plt.colorbar(im,ax=ax,shrink=.75)
    fig,ax=plt.subplots(figsize=(10,4));z=counts[counts.history_seconds==5];heat(ax,z,'n_anchors','5초 primary 상태별 표본 수 (anchors)');save('state_counts_heatmap',fig,z)
    fig,axs=plt.subplots(1,3,figsize=(19,4));z=primary[(primary['lambda']==1)&primary.model.isin(['A','R','RM'])]
    for ax,m in zip(axs,['A','R','RM']):heat(ax,z[z.model==m],'gain',f'{m}, λ=1: MAE gain (W)',True)
    save('primary_gain_heatmap',fig,z)
    fig,axs=plt.subplots(2,3,figsize=(17,9))
    for ax,c in zip(axs.flat,CELLS):
        z=primary[primary.cell==c]
        for m in ['R','RM']:
            d=z[z.model==m].sort_values('lambda');p=z[z.model=='P'];ax.plot([0]+d['lambda'].tolist(),[0]+d.gain.tolist(),marker='o',label=m)
        a=z[z.model=='A'].iloc[0];ax.axhline(a.gain,ls=':',color='gray',label='A λ=1');ax.axhline(0,color='black',lw=.7)
        ax.set_title(f'{LABEL[c]} | {int(a.valid_runs)} runs, {int(a.n_anchors)} anchors');ax.set_xlabel('λ');ax.set_ylabel('MAE gain (W)');ax.legend()
    save('cell_lambda_gain',fig,primary)
    fig,ax=plt.subplots(figsize=(14,6));z=dc[(dc.history_seconds==5)&(dc['lambda']==1)]
    x=np.arange(6)
    for i,m in enumerate(['A','R','RM']):
        d=z[z.model==m].set_index('cell').reindex(CELLS);ax.bar(x+(i-1)*.24,d.gain_contribution,width=.24,label=m)
    labels=[f'{LABEL[r.cell]}: {r.valid_cell_runs} runs/{r.n_anchors} anchors' for r in z[z.model=='R'].itertuples()]
    ax.set_xticks(x,[LABEL[c] for c in CELLS]);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('전체 MAE gain 기여도 (W)');ax.set_title('λ=1 | 전체 23 runs 동일 가중; cell이 없는 run 기여도 0\n'+', '.join(labels[:3])+'\n'+', '.join(labels[3:]),fontsize=10);ax.legend();save('gain_contributions',fig,z)
    fig,ax=plt.subplots(figsize=(12,6));z=day[(day.history_seconds==5)&(day.cell=='low_large')&(day['lambda']==1)&day.model.isin(['A','R','RM'])]
    dates=sorted(z.collection_day.unique())
    for m in ['A','R','RM']:
        d=z[z.model==m].set_index('collection_day').reindex(dates);ax.plot(dates,d.gain,marker='o',label=m)
    ax.axhline(0,color='black',lw=.7);ax.set_ylabel('MAE gain (W)');ax.set_title('5초 저변동→큰 변화, λ=1\n'+', '.join(f'{r.collection_day}: {r.valid_runs} runs/{r.n_anchors} anchors' for r in z[z.model=='R'].itertuples()));ax.legend();save('low_large_day_gain',fig,z)
    fig,axs=plt.subplots(2,3,figsize=(17,9));z=eff[(eff.scope=='state')&(eff.history_seconds==5)&eff.cell.isin(CELLS)]
    for ax,c in zip(axs.flat,CELLS):
        d=z[z.cell==c].sort_values('lambda');ax.plot(d['lambda'],d.mae_effect,marker='o');ax.axhline(0,color='black',lw=.7);ax.set_xlabel('λ');ax.set_ylabel('MAE_R - MAE_RM (W)');r=d.iloc[0];ax.set_title(f'{LABEL[c]} | {int(r.valid_runs)} runs/{int(r.n_anchors)} anchors')
    save('metadata_effect',fig,z)
    fig,axs=plt.subplots(1,3,figsize=(18,6));z=s[(s['lambda']==1)&s.cell.isin(CELLS)&s.model.isin(['A','R','RM'])]
    for ax,m in zip(axs,['A','R','RM']):
        for h in [5,1]:
            d=z[(z.model==m)&(z.history_seconds==h)].set_index('cell').reindex(CELLS);ax.plot(x,d.gain,marker='o',label=f'{h}초');
            for i,r in enumerate(d.itertuples()):ax.annotate(f'{int(r.valid_runs)}r/{int(r.n_anchors)}a',(i,r.gain),xytext=(0,8 if h==5 else -14),textcoords='offset points',ha='center',fontsize=7)
        ax.set_xticks(x,[LABEL[c] for c in CELLS],rotation=30,ha='right');ax.set_ylabel('MAE gain (W)');ax.axhline(0,color='black',lw=.7);ax.set_title(m+' λ=1; r=runs, a=anchors');ax.legend()
    save('sensitivity_comparison',fig,z)
    fig,axs=plt.subplots(2,1,figsize=(11,9));z=s[(s.history_seconds==5)&s.cell.str.startswith('first_departure')]
    for m in ['A','R','RM']:
        d=z[(z.model==m)&(z['lambda']==1)].sort_values('cell');axs[0].plot(np.arange(1,6)*.2,d.gain,marker='o',label=m)
    axs[0].set_ylabel('MAE gain (W)');axs[0].axhline(0,color='black',lw=.7);axs[0].legend();axs[0].set_title('5초 저변동→큰 변화: 현재값 대비 큰 차이의 최초 관측 bin')
    d=z[z.model=='P'].sort_values('cell');axs[1].bar(np.arange(1,6)*.2,d.n_anchors,width=.12)
    for i,r in enumerate(d.itertuples()):axs[1].text((i+1)*.2,r.n_anchors,f'{int(r.valid_runs)} runs\n{int(r.n_anchors)} anchors',ha='center',va='bottom')
    axs[1].set_ylim(0,max(1,d.n_anchors.max())*1.25);axs[1].set_ylabel('anchors');axs[1].set_xlabel('발행 시각 이후 bin 끝 시각 (s)');save('first_departure',fig,z)
    cases=[]
    for c in CELLS:
        g=base[base.cell_5s==c].sort_values(['run_id','segment','bin'])
        if g.empty:continue
        r=g.iloc[0];fig,ax=plt.subplots(figsize=(12,5));t=np.arange(-24,6)*.2;power=np.array([r[f'x_{j}'] for j in range(-24,6)])
        ax.plot(t,power,'o-',ms=3,color='black',label='실제 완료 bin');ax.axvline(0,ls='--',color='gray',label='발행 시각')
        for m,color,marker in [('P','tab:blue','s'),('A','tab:orange','^'),('R','tab:green','D'),('RM','tab:red','x')]:ax.scatter([1],[r.current_w if m=='P' else r[m]],s=80,marker=marker,color=color,label=m+' 저장 예측 (λ=1)' if m!='P' else 'P')
        ax.set_xlabel('발행 시각 대비 완료 bin 끝 시각 (s)');ax.set_ylabel('8-GPU 합산 전력 (W)');ax.set_title(f'{LABEL[c]} | 사전순 첫 key, 1 run / 1 anchor\n{r.run_id}, segment={r.segment}, bin={r.bin}, 발행={r.issuance_s:.1f}s');ax.legend(loc='best')
        data=pd.DataFrame({'relative_bin_end_s':t,'actual_w':power});data['run_id']=r.run_id;data['segment']=r.segment;data['bin']=r.bin
        for m in ['P','A','R','RM']:data[m+'_target_prediction_w']=r.current_w if m=='P' else r[m]
        name='case_'+c;save(name,fig,data);cases.append(dict(cell=c,run_id=r.run_id,segment=int(r.segment),bin=int(r.bin),issuance_s=r.issuance_s,figure=name))
    pd.DataFrame(cases).to_csv(out/'case_selection.csv',index=False)
    write=lambda name,x:(out/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
    write('render_verification.json',dict(figures=checks,case_rule='lexicographic first (run_id, segment, bin) within each primary cell',all_figures_exist=all((figdir/f'{n}.png').exists() and (figdir/f'{n}.svg').exists() for n in figs)))
    def table(d,cols=None):
        if cols:d=d[cols]
        return d.to_markdown(index=False,floatfmt='.3f',missingval='N/A').replace('nan','N/A')
    def read(name):return pd.read_csv(out/(name+'.csv'))
    def rows(h,cell):return s[(s.history_seconds==h)&(s.cell==cell)&(s['lambda']==1)&s.model.isin(['A','R','RM'])]
    lines=['# AIDC 과거·미래 상태 진단','',f'분석 완료: **{base.run_id.nunique()}개 run · {len(base):,} anchors**, 신규 fit **0회**, 신규 모델 inference **0회**. 개발 날짜 2025-09-16·17·20의 기존 leave-day-out OOF만 사용했다. 2025-09-22 전력 행은 로드하거나 평가하지 않았다.','', '## 1. 핵심 판정','']
    low=rows(5,'low_large');high=rows(5,'high_large')
    for m in ['A','R','RM']:
        r=low[low.model==m].iloc[0];hh=high[high.model==m].iloc[0]
        lines.append(f'- {m}, 원래 예측 λ=1: 5초 저변동→큰 변화 gain **{r.gain:.3f} W** ({int(r.valid_runs)} 유효 runs, {int(r.n_anchors)} anchors); 고변동→큰 변화 gain {hh.gain:.3f} W. 양수는 Persistence 대비 조건부 MAE 개선이다.')
    # Answer all seven requested judgments using the same saved numeric tables.
    for m in ['A','R','RM']:
        v=dc[(dc.history_seconds==5)&(dc.model==m)&(dc['lambda']==1)].set_index('cell').gain_contribution
        dd=day[(day.history_seconds==5)&(day.cell=='low_large')&(day.model==m)&(day['lambda']==1)]
        lines.append(f'- {m}: 큰 변화 cell의 전체 gain 기여도는 저변동 {v.low_large:.3f} W, 고변동 {v.high_large:.3f} W다. 고변동 큰 변화의 기여도가 더 크지만 저변동에도 조건부 이득이 있다. 저변동→큰 변화 날짜별 양수 gain은 {int((dd.gain>0).sum())}/{len(dd)}일이다.')
    e5=eff[(eff.scope=='state')&(eff.history_seconds==5)&(eff['lambda']==1)].set_index('cell')
    e1=eff[(eff.scope=='state')&(eff.history_seconds==1)&(eff['lambda']==1)].set_index('cell')
    lines += [f'- Metadata: 5초 저변동→큰 변화 MAE 추가 효과는 {e5.loc["low_large","mae_effect"]:.3f} W, 저변동→작은 변화는 {e5.loc["low_small","mae_effect"]:.3f} W다. 저변동→작은 변화에서 RM−R 예측 변화량 절댓값 차이는 {-e5.loc["low_small","predicted_abs_delta_effect"]:.3f} W다. 양수이면 불필요한 보정량이 커진 관찰이다.']
    for m in ['R','RM']:
        r=primary[(primary.model==m)&(primary['lambda']==1)].set_index('cell')
        q=primary[(primary.model==m)&(primary['lambda']==.25)].set_index('cell')
        lines.append(f'- {m} 축소 λ=1→0.25: 저변동→작은 변화 gain {r.loc["low_small","gain"]:.3f}→{q.loc["low_small","gain"]:.3f} W, 고변동→작은 변화 {r.loc["high_small","gain"]:.3f}→{q.loc["high_small","gain"]:.3f} W; 저변동→큰 변화 {r.loc["low_large","gain"]:.3f}→{q.loc["low_large","gain"]:.3f} W, 고변동→큰 변화 {r.loc["high_large","gain"]:.3f}→{q.loc["high_large","gain"]:.3f} W. 작은 변화 손해와 큰 변화 이득을 모두 줄였다.')
    one=rows(1,'low_large')
    lines += [f'- 1초 저변동→큰 변화 gain: '+', '.join(f'{r.model} {r.gain:.3f} W' for r in one.itertuples())+f'. 저변동에서도 이득을 관찰했다는 해석은 유지되지만 metadata 효과는 5초 {e5.loc["low_large","mae_effect"]:.3f} W에서 1초 {e1.loc["low_large","mae_effect"]:.3f} W로 부호가 달라진다.', f'- 표본 제한: 5초 핵심 cell은 {int(low.iloc[0].n_anchors)} anchors/{int(low.iloc[0].valid_runs)} runs 중 {int(low.iloc[0].small_n_runs)} runs가 small_n; 1초는 {int(one.iloc[0].n_anchors)} anchors/{int(one.iloc[0].valid_runs)} runs 중 {int(one.iloc[0].small_n_runs)} runs가 small_n이다. 날짜·장치·workload 교락으로 인과적 분리와 독립 일반화 판단은 제한된다.', '', '**판정: 저변동 상태에서도 미래 큰 endpoint 변화의 조건부 이득 관찰. 큰 변화의 전체 양수 기여도는 고변동에 더 많이 분포하며, metadata 효과는 날짜·상태 정의에 따라 혼합이다. 전체 MAE에서는 고정 양수 λ와 A 모두 Persistence 개선 근거가 없다.**', '']
    lines+=['','날짜별·상태 정의별 결과는 아래 수치로 함께 판단한다. 조건부 cell 평균의 run 집합은 서로 다를 수 있으므로 두 평균의 차이를 순수한 상태 효과로 해석하지 않는다.','',table(low,['model','mae','gain','valid_runs','n_anchors','small_n_runs','improved_runs','tied_runs','worsened_runs','direction_match']), '','## 2. 입력 provenance와 시간 정렬','', 'PROJECT_ROOT는 `/home/work/AIDC`이며 Git 저장소가 아니고 적용 AGENTS.md는 없다. 기존 산출물을 보존했고 실제 경로·SHA-256은 `resolved_paths.json`, `input_manifest.json`에 저장했다. 기존 shrinkage의 read-only 검증을 재사용하여 fit 완료 manifest와 anchor·모델·weight·prediction hash를 확인했다. 모델 파일은 hash만 읽고 모델을 로드하지 않았다. 원본 preprocessing의 센서 표본 산술 평균으로 생성된 완료 bin `prepared.parquet`에서 개발 run_id만 Arrow filter로 읽었다. 보류 날짜 전력 행은 materialize하지 않았다.','',table(read('provenance'),['model','fold','train_days','evaluation_day','train_runs','train_anchors','n_runs','n_anchors']),'','현재 x[k], target x[k+5]를 key `(run_id, segment, bin)`로 결합했다. 모든 offset -24…5의 존재·유한값·시각·동일 segment를 확인했고 보간·anchor 제외는 없다. 5초 history는 x[k-24]…x[k] 25개 완료 bin, 1초는 x[k-4]…x[k] 5개다. 다섯 bin의 측정 구간 합은 1초이고 끝 시각끼리의 차이는 0.8초다. Target은 `(t+0.8,t+1.0]` 8-GPU 합산 전력의 완료 bin 평균이다. 미래 bin은 사후 분류·설명에만 사용했다.','', '## 3. 고정 상태 정의와 threshold','', '과거 range=max−min. Fold별 기존 cap300 training anchor 전체의 비가중 NumPy linear q50 이하가 저변동, 초과가 고변동이다. 5초가 primary, 1초가 사전 고정 sensitivity다. 저변동은 상대적 구분이며 무변화·정상 운전이라는 뜻이 아니다. 고변동 range에는 이미 끝난 변화와 진동도 포함될 수 있다. 미래 endpoint 절댓값은 training q50 이하 작은 변화, q50 초과 q90 이하 중간 변화, q90 초과 큰 변화로 상호 배타적으로 구분한다.','',table(read('thresholds')),'','## 4. 상태별 표본 수와 구성','',table(counts),'', 'Run별 비중·small_n은 `state_counts_per_run.csv`, 날짜×hardware×workload 구성은 `state_composition_day_hardware_workload.csv`에 저장했다. small_n은 해당 run-cell n<30의 기술적 표시이고 통계적 충분성 기준이 아니다. 빈 cell의 성능은 N/A이며 숨기거나 병합하지 않았다.','',table(read('state_composition_day_hardware_workload').query('history_seconds == 5 and n_anchors > 0')),'','## 5. Primary 여섯 cell 성능','', '각 지표는 run-cell에서 먼저 계산하고 비어 있지 않은 run 간 동일 가중 평균했다. RMSE·p95·p99도 run별 값의 평균이다. p95는 n≥20, p99는 n≥100에서만 계산하며 유효 run 수를 병기했다. Signed bias=prediction−actual. 방향 일치율은 미래 큰 변화에서 sign(prediction−current)=sign(actual−current)의 비율이며 0 예측 변화는 불일치다. λ=0은 P로 한 번만 집계했다. R/RM은 λ=0.25·0.5·0.75·1을 모두 보고하고 A는 기존 예측만 사용한다.','',table(primary),'','큰 변화 상승·하강은 `large_direction_metrics.csv`와 전체 `metrics_per_run_state.csv`에서 확인한다. 원래 예측 λ=1의 세부 값:','',table(s[(s.history_seconds==5)&s.cell.str.contains('large_rising|large_falling')&(s['lambda']==1)]),'','## 6. 전체 MAE gain 기여도','', '각 run에서 cell gain 합계를 해당 run의 전체 anchor 수로 나눈 뒤 모든 개발 run을 동일 가중 평균했다. 빈 cell run은 기여도 0이다. 여섯 cell 합은 전체 gain과 일치하며, 상승·하강 및 1초 sensitivity는 여기에 다시 더하지 않는다.','',table(dc[dc.history_seconds==5]),'', '조건부 개선과 전체 기여도는 다른 집계다. λ=1 저변동·고변동의 기여도 합:','']
    z=dc[(dc.history_seconds==5)&(dc['lambda']==1)].copy();z['past_state']=z.cell.str.split('_').str[0]
    lines += [table(z.groupby(['model','past_state']).gain_contribution.sum().reset_index()),'', '## 7. Metadata 추가 효과','', '동일 cell·λ에서 MAE_R−MAE_RM이 양수이면 metadata 추가로 MAE가 줄었다. Bias 차이와 예측 변화량 차이는 개선의 부호가 아니다. λ=0은 양쪽 모두 P이므로 metadata 효과가 정확히 0이며 중복 성능 행을 만들지 않았다.','',table(eff[(eff.scope=='state')&(eff.history_seconds==5)&eff.cell.isin(['whole']+CELLS)],['cell','lambda','mae_effect','signed_bias_effect','predicted_abs_delta_effect','n_anchors','valid_runs','small_n_runs']),'','저변동→큰 변화와 저변동→작은 변화의 날짜별 효과:','',table(eff[(eff.scope=='day_state')&(eff.history_seconds==5)&eff.cell.isin(['low_large','low_small'])],['collection_day','cell','lambda','mae_effect','predicted_abs_delta_effect','n_anchors','valid_runs','small_n_runs']),'','날짜와 hardware/workload는 교락되어 있다. 동일 표본의 R/RM 차이는 metadata 추가 효과를 기술하지만 특정 날짜·장치·workload의 인과 효과를 분리하지 못한다.','', '## 8. 날짜 일관성과 shrinkage','',table(day[(day.history_seconds==5)&(day.cell=='low_large')]),'', '축소 효과는 λ=1과 같은 모델·cell의 다른 고정 λ를 비교한다. 정상·변화라는 사후 label로 계수를 선택하거나 cell별 최저 계수를 결합하지 않았다.','']
    shrink=primary[primary.model.isin(['R','RM'])].merge(primary[primary['lambda']==1][['cell','model','mae','gain']].rename(columns={'mae':'mae_lambda1','gain':'gain_lambda1'}),on=['cell','model'])
    shrink['mae_reduction_vs_lambda1']=shrink.mae_lambda1-shrink.mae
    lines += [table(shrink,['cell','model','lambda','gain','gain_lambda1','mae_reduction_vs_lambda1','valid_runs','n_anchors']),'','## 9. 1초 sensitivity','',table(s[(s.cell.isin(['low_large','high_large']))&(s['lambda']==1)]),'', '완전한 여섯 cell·모든 고정 λ 비교는 `sensitivity_comparison.csv`에 있다. 상태 정의가 달라지면 membership·유효 run 집합도 달라지므로 지표 차이를 동일 표본의 순수 효과로 단정하지 않는다.','', '## 10. 미래 경로 보조 진단','', '저변동→큰 변화에서 first_large_departure는 현재값 대비 endpoint training q90을 처음 초과한 미래 bin j∈1…5다. 이는 실제 물리적 전이 시작이나 보장된 lead time이 아니다. 보조 진단으로 계수·threshold를 선택하지 않았다.','',table(s[(s.history_seconds==5)&s.cell.str.startswith('first_departure')]),'', '## 11. 검증·한계와 후속 방향','']
    verification=json.loads((out/'verification.json').read_text())
    lines += [table(pd.DataFrame([dict(check=k,result=str(v)) for k,v in verification.items()])), '', '이 분석은 개발 OOF를 반복 사용한 사후 기술 진단이며 독립 일반화 평가·bootstrap·유의성 검정이 아니다. 저변동→큰 변화의 개선은 전이 시작 시각 예측을 입증하지 않는다. 현재 bin에 변화 일부가 포함되거나 threshold 아래 전조가 있을 수 있다. 반대로 개선이 없다고 전력 예측 자체가 불가능하다고 판단하지 않는다. 희소 cell의 날짜별 run 수와 small_n을 함께 보아야 하고, 날짜·hardware/workload 교락으로 분리 판단이 제한된다. Gate 학습 가능성이나 실행 가능한 조건별 계수를 주장하지 않는다.','', '후속 개발은 별도로 설계한 과거 정보만의 진단과 독립 검증으로 진행할 수 있다. 이번 실행에서는 새 모델·gate·추가 threshold·보류 날짜 평가·제어·MPC·계통 시뮬레이션을 수행하지 않았다.','', '## 12. 그림과 재현','']
    for name in figs:lines += [f'![{name}](figures/{name}.png)','']
    lines += ['각 그림의 PNG·SVG·plotting CSV를 저장했다. 여섯 사례는 각 primary cell의 sample key 사전순 첫 anchor다. 실제 연속 완료 bin 선과 target 시각의 저장 예측 점을 구분했다. 불연속 평가 anchor 예측을 연결하지 않았다. `render_verification.json`에서 파일 존재와 텍스트의 canvas 경계 검사를 기록했다.','', '`README.md`의 smoke·nohup launch·status·resume·report-only 명령으로 재현한다. Input/code/config hash가 모두 같을 때만 resume를 허용하며 중복 프로세스·분석 lock으로 재실행을 막는다. 실행 상태는 `pipeline.log`, `pipeline.pid`, `process_identity.json`, `progress.json`, `completion_status.json`, `exit_code.txt`; 시간은 `timing.json`, `report_only_timing.json`에 저장한다.']
    md='\n'.join(lines);(out/'report_ko.md').write_text(md)
    html=markdown.markdown(md,extensions=['tables','fenced_code'])
    (out/'report_ko.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>AIDC 과거·미래 상태 진단</title><style>body{font-family:Arial,"NanumGothic",sans-serif;margin:28px;line-height:1.6}table{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}td,th{border:1px solid #ccc;padding:5px;white-space:nowrap}img{max-width:100%;height:auto}h2{margin-top:32px}</style><body>'+html+'</body></html>')
    assert not re.search(r'\\[|!\[\]]',md)
    for n in figs:assert (figdir/(n+'.png')).is_file()
